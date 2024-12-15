import torch
import torch.nn as nn


class VAE(nn.Module):

    def __init__(self, in_dim, hidden_dim, latent_dim, conditional=False, condition_dim=None):

        super().__init__()

        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.conditional = conditional

        if self.conditional and condition_dim is not None:
            self.input_dim = self.in_dim + self.condition_dim
            self.dec_dim = self.latent_dim + self.condition_dim
        else:
            self.input_dim = self.in_dim
            self.dec_dim = self.latent_dim
        self.enc_MLP = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ELU())
        self.linear_means = nn.Linear(hidden_dim, latent_dim)
        self.linear_log_var = nn.Linear(hidden_dim, latent_dim)
        self.dec_MLP = nn.Sequential(
            nn.Linear(self.dec_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self.in_dim))

    def forward(self, gt, condition=None):
        if self.conditional and condition is not None:
            input = torch.cat((gt, condition), dim=-1)
        else:
            input = gt
        h = self.enc_MLP(input)
        mean = self.linear_means(h)
        log_var = self.linear_log_var(h)
        z = self.reparameterize(mean, log_var).to(gt.device).to(gt.dtype)
        if self.conditional and condition is not None:
            z = torch.cat((z, condition), dim=-1)
        reconstructed_gt = self.dec_MLP(z)
        recon_loss, KLD = self.loss_fn(reconstructed_gt, gt, mean, log_var)
        return reconstructed_gt, recon_loss, KLD

    def loss_fn(self, recon_x, x, mean, log_var):
        recon_loss = torch.sum((recon_x - x) ** 2, dim=1)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
        return recon_loss, KLD

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def inference(self, z, c=None):
        if self.conditional and c is not None:
            z = torch.cat((z, c), dim=-1)
        recon_x = self.dec_MLP(z)
        return recon_x