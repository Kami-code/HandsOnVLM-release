import os
import argparse
import random

import torch
import torch.optim
import torch.nn.parallel
import numpy as np
import wandb

from hoi_forecast.model.optimizer import get_optimizer
from hoi_forecast.model import modelio
from hoi_forecast.model.trainer import epoch_evaluate, epoch_pass
from hoi_forecast.options import netsopts, expopts
from hoi_forecast.dataset.dataloader import get_epic_hoi_dataloader_by_name
from hoi_forecast.utils.const import observation_frames_num, anticipation_frames_num
from hoi_forecast.model.build_model import get_hoi_forecast_origin_model


def main(args):
    # Initialize randoms seeds
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    ek_version = args.ek_version
    batch_size = args.batch_size

    model_name = "hoi_forecast_original"
    wandb_name = f"{model_name}_{args.exp_id}"
    wandb.init(project="perfect-hoi-forcast-newmodelarch", entity="dexsuite", name=wandb_name, config=args, #$id=wandb_id,
               dir="/ocean/projects/cis240031p/cbao/output_dir", resume="allow")

    model = get_hoi_forecast_origin_model(num_frames_input=observation_frames_num,
                                          num_frames_output=anticipation_frames_num,
                                          lambda_obj=args.lambda_obj,
                                          lambda_traj=args.lambda_traj,
                                          lambda_obj_kl=args.lambda_obj_kl,
                                          lambda_traj_kl=args.lambda_traj_kl,
                                          lambda_last_hand=args.lambda_last_hand,
                                          )

    if args.use_cuda and torch.cuda.is_available():
        print("Using {} GPUs !".format(torch.cuda.device_count()))
        model.cuda()

    start_epoch = 0
    device = torch.device('cuda') if torch.cuda.is_available() and args.use_cuda else torch.device('cpu')

    initial_epoch = wandb.run.step
    print("wandb_initial_epoch = ", initial_epoch)

    checkpoint_dir = os.path.join(args.host_folder, args.exp_id)
    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_latest.pth.tar")
    print("ckpt_path = ", ckpt_path)
    if os.path.exists(ckpt_path):
        start_epoch = modelio.load_checkpoint(model, resume_path=ckpt_path, strict=False, device=device) + 1
        print("Loaded checkpoint from epoch {}, starting from there".format(start_epoch))
    num_workers = 0

    train_loader = get_epic_hoi_dataloader_by_name(ek_version=ek_version, split="train", batch_size=batch_size, num_workers=num_workers)
    print("training dataset size: {}".format(len(train_loader.dataset)))
    optimizer, scheduler = get_optimizer(args, model=model, train_loader=train_loader)
    traj_train_loader_ek100 = get_epic_hoi_dataloader_by_name(ek_version="ek100", split="train", batch_size=batch_size, num_workers=num_workers)
    affordance_val_loader_ek100 = get_epic_hoi_dataloader_by_name(ek_version="ek100", split="eval", batch_size=batch_size, num_workers=num_workers)
    traj_val_loader_ek100 = get_epic_hoi_dataloader_by_name(ek_version="ek100", split="validation", batch_size=batch_size, num_workers=num_workers)
    print("affordance_val_loader_ek100 dataset size: {}".format(len(affordance_val_loader_ek100.dataset)))
    print("traj_val_loader_ek100 dataset size: {}".format(len(traj_val_loader_ek100.dataset)))

    for epoch in range(start_epoch, args.epochs):
        if args.evaluate or (epoch + 1) % args.test_freq == 0:
            with torch.no_grad():
                val_results = {}
                train_traj_metrics_ek100 = epoch_evaluate(
                    loader=traj_train_loader_ek100,
                    model=model,
                    epoch=epoch,
                    phase='traj',
                    num_samples=args.num_samples, visualize=True)
                traj_metrics_ek100 = epoch_evaluate(
                    loader=traj_val_loader_ek100,
                    model=model,
                    epoch=epoch,
                    phase='traj',
                    num_samples=args.num_samples, visualize=True)
                for k, v in traj_metrics_ek100.items():
                    val_results[f"validation_ek100/{k}"] = v
                for k, v in train_traj_metrics_ek100.items():
                    val_results[f"train_ek100/{k}"] = v

                print("validation results: ", val_results)
                wandb.log(val_results, step=epoch)
        if not args.evaluate:
            print("Using lr {}".format(optimizer.param_groups[0]["lr"]))
            epoch_pass(
                loader=train_loader,
                model=model,
                phase='train',
                optimizer=optimizer,
                epoch=epoch,
                scheduler=scheduler)
        save_exp_path = os.path.join(args.host_folder, args.exp_id)
        os.makedirs(save_exp_path, exist_ok=True)
        modelio.save_checkpoint(
            {
                "epoch": epoch + 1,
                # "network": args.network,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            checkpoint=os.path.join(args.host_folder, args.exp_id),
            filename=f"checkpoint_latest.pth.tar")
        if not args.evaluate:
            if (epoch + 1 - args.warmup_epochs) % args.snapshot == 0:
                print(f"save epoch {epoch} checkpoint to {os.path.join(args.host_folder, args.exp_id)}")
                modelio.save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    checkpoint=os.path.join(args.host_folder, args.exp_id),
                    filename=f"checkpoint_{epoch}.pth.tar")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HOI Forecasting")
    netsopts.add_nets_opts(parser)
    netsopts.add_train_opts(parser)
    expopts.add_exp_opts(parser)
    args = parser.parse_args()

    if args.use_cuda and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        args.batch_size = args.batch_size * num_gpus
        args.lr = args.lr * num_gpus

    if args.traj_only: assert args.evaluate, "evaluate trajectory on validation set must set --evaluate"
    main(args)
    print("All done !")