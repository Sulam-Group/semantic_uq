import argparse
import os

import torch
import torch.distributed as distributed
import torch.nn.functional as F
import wandb
from monai.data import DataLoader, DistributedSampler
from monai.inferers import sliding_window_inference
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from configs import Config
from configs import Constants as C
from configs import get_config
from datasets import AbdomenAtlas, get_dataset
from model import Denoiser
from utils import Monitor

monitor = Monitor()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--from_latest", action="store_true", default=False)
    parser.add_argument("--dist", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=1e-04)
    parser.add_argument("--weight_decay", type=float, default=1e-05)
    parser.add_argument("--workdir", type=str, default=C.workdir)
    return parser.parse_args()


def pinball_loss(
    input: torch.Tensor, target: torch.Tensor, alpha: float, reduction: str = "none"
):
    assert alpha > 0 and alpha < 1, f"alpha must be in (0,1), got {alpha}"
    assert input.size() == target.size(), (
        f"input and target must have the same size, got {input.size()} and"
        f" {target.size()}"
    )

    error = input - target
    loss = error.abs() * (alpha * (error < 0) + (1 - alpha) * (error > 0))

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"reduction must be one of [none, mean, sum], got {reduction}")


def loss_fn(config: Config, output: torch.Tensor, target: torch.Tensor):
    q_lo, q_hi = config.data.q_lo, config.data.q_hi
    loss_lo = pinball_loss(output[:, 0], target[:, 0], q_lo, reduction="sum")
    loss_mse = F.mse_loss(output[:, 1], target[:, 0], reduction="sum")
    loss_hi = pinball_loss(output[:, 2], target[:, 0], q_hi, reduction="sum")
    return loss_lo + loss_mse + loss_hi


def train_epoch(
    config: Config,
    model: Denoiser,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    dist: bool,
):
    monitor.zero()
    model.train()

    rank, world_size = 0, 1
    if dist:
        rank = distributed.get_rank()
        world_size = distributed.get_world_size()

    for i, data in enumerate(tqdm(dataloader)):
        image = data["image"].to(device)

        if config.data.task == "denoising":
            measurement = image + config.data.sigma * torch.randn_like(image)
        elif config.data.task == "reconstruction":
            measurement = data["fbp"].to(device)
        else:
            raise ValueError(f"Invalid task: {config.data.task}")

        optimizer.zero_grad()
        output = model(measurement)
        loss = loss_fn(config, output, image)
        loss.backward()

        optimizer.step()

        if rank == 0:
            monitor.update(
                {"loss": world_size * loss.cpu().item()},
                num_samples=world_size * image.size(0),
            )

        log_step = 10
        if (i + 1) % log_step == 0 and rank == 0:
            monitor.log(prefix="train")


@torch.no_grad()
def evaluate(
    config: Config, model: Denoiser, dataloader: DataLoader, device: torch.device
):
    monitor.zero()
    model.eval()

    for i, data in enumerate(tqdm(dataloader)):
        image = data["image"].to(device)

        perturbed = image + config.data.sigma * torch.randn_like(image)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = sliding_window_inference(perturbed, (96, 96, 96), 1, model)
            output_mse = output[:, [1]]
            loss = F.mse_loss(output_mse, image, reduction="sum")

        monitor.update(
            {"loss": loss.cpu().item()},
            num_samples=image.size(0),
            increase_global_samples=False,
        )

        if (i + 1) % 20 == 0:
            break

    monitor.log(prefix="val")


def main(args):
    config_name = args.config
    from_latest = args.from_latest
    dist = args.dist
    workdir = args.workdir

    config = get_config(config_name)

    rank = 0
    if dist:
        distributed.init_process_group(backend="nccl")
        rank = distributed.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    with_fbp = config.data.task == "reconstruction"
    dataset = AbdomenAtlas(config, with_fbp=with_fbp, workdir=workdir)
    val_dataset = get_dataset(config, with_fbp=with_fbp, workdir=workdir)

    sampler = None
    if dist:
        sampler = DistributedSampler(dataset, even_divisible=False, shuffle=True)

    dataloader = DataLoader(
        dataset, batch_size=1, sampler=sampler, shuffle=sampler is None, num_workers=4
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)

    if from_latest:
        print("Loading model from the latest checkpoint")
        model = Denoiser.from_pretrained(config, workdir=workdir, device=device)
    else:
        model = Denoiser(config, device=device)

    if dist:
        model = DistributedDataParallel(model, device_ids=[device])
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if from_latest:
        weights_dir = os.path.join(workdir, "weights", config.results_name)
        with open(os.path.join(weights_dir, "latest.txt"), "r") as f:
            latest = f.read().strip()
        checkpoint = torch.load(os.path.join(weights_dir, latest), map_location="cpu")
        optimizer.load_state_dict(checkpoint["optimizer"])

    if rank == 0:
        wandb.init(project="abdomen_denoising", config=vars(config))

    weights_dir = os.path.join(workdir, "weights", config.results_name)
    os.makedirs(weights_dir, exist_ok=True)

    max_epochs = 100
    for t in range(max_epochs):
        train_epoch(config, model, optimizer, dataloader, device, dist)

        if rank == 0:
            evaluate(config, model, val_dataloader, device)

            state_dict = model.module.state_dict() if dist else model.state_dict()
            torch.save(
                {"model": state_dict, "optimizer": optimizer.state_dict()},
                os.path.join(weights_dir, f"model_{t+1}.pt"),
            )
            with open(os.path.join(weights_dir, "latest.txt"), "w") as f:
                f.write(f"model_{t+1}.pt")

    if dist:
        distributed.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    main(args)
