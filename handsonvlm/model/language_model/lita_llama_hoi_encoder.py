import torch.nn as nn

from handsonvlm.model.language_model.lita_llama_encoder import LitaLlamaForCausalLM_encoder


class LitaLlamaForCausalLM_hoi_encoder(LitaLlamaForCausalLM_encoder):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.coord_dim = 64
        self.bbox_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_dim // 2),
            nn.ELU(inplace=True),
            nn.Linear(self.coord_dim // 2, self.coord_dim),
            nn.ELU()
        )
        self.feat_fusion = nn.Sequential(
            nn.Linear(1024 + self.coord_dim, 1024),
            nn.ELU(inplace=True))
        self.downproject = nn.Linear(1024, 1024)

        self.added_modules.append(self.bbox_to_feature)
        self.added_modules.append(self.feat_fusion)
        self.added_modules.append(self.downproject)

        self.extra_kwargs['downproject'] = self.downproject
        self.extra_kwargs['bbox_to_feature'] = self.bbox_to_feature
        self.extra_kwargs['feat_fusion'] = self.feat_fusion
