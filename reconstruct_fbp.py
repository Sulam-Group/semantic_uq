import argparse
import os

import numpy as np
import odl
import torch
import torch.distributed as distributed
from monai.data import DataLoader, DistributedSampler, list_data_collate
from odl.contrib.torch import OperatorModule
from tqdm import tqdm

from configs import Config
from configs import Constants as C
from configs import get_config
from datasets import AbdomenAtlas, get_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--atlas", action="store_true", default=False)
    parser.add_argument("--dist", action="store_true", default=False)
    parser.add_argument("--workdir", type=str, default=C.workdir)
    return parser.parse_args()


def helical_geometry(config: Config, space: odl.DiscretizedSpace):
    corners = space.domain.corners()[:, :2]
    rho = np.max(np.linalg.norm(corners, axis=1))

    offset_along_axis = space.partition.min_pt[2]
    pitch = space.partition.extent[2] / config.data.num_turns

    min_side = min(space.partition.cell_sides[:2])
    omega = np.pi / min_side

    rs = float(config.data.src_radius)
    if rs <= rho:
        raise ValueError(
            "source too close to the object, resulting in "
            "infinite detector for full coverage"
        )
    rd = float(config.data.det_radius)
    r = rs + rd
    w = 2 * rho * (rs + rd) / rs

    rb = np.hypot(r, w / 2)
    num_px_horiz = 2 * int(np.ceil(w * omega * r / (2 * np.pi * rb))) + 1

    h_axis = (
        pitch
        / (2 * np.pi)
        * (1 + (-rho / config.data.src_radius) ** 2)
        * (1 * np.pi / 2.0 - np.arctan(-rho / config.data.src_radius))
    )
    h = 2 * h_axis * (rs + rd) / rs

    min_mag = r / rs
    dh = 0.5 * space.partition.cell_sides[2] * min_mag
    num_px_vert = int(np.ceil(h / dh))

    det_min_pt = [-w / 2, -h / 2]
    det_max_pt = [w / 2, h / 2]
    det_shape = [
        np.clip(num_px_horiz, None, config.data.max_det_shape[0]),
        np.clip(num_px_vert, None, config.data.max_det_shape[1]),
    ]

    max_angle = 2 * np.pi * config.data.num_turns
    num_angles = int(np.ceil(max_angle * omega * rho / np.pi * r / (r + rho)))
    num_angles = np.clip(num_angles, None, config.data.max_num_angles)

    angle_partition = odl.uniform_partition(0, max_angle, num_angles)
    det_partition = odl.uniform_partition(det_min_pt, det_max_pt, det_shape)
    return odl.tomo.ConeBeamGeometry(
        angle_partition,
        det_partition,
        config.data.src_radius,
        config.data.det_radius,
        offset_along_axis=offset_along_axis,
        pitch=pitch,
    )


def get_operator(config: Config, image: torch.Tensor):
    x, y, z = image.shape[-3:]

    reco_space = odl.uniform_discr(
        min_pt=[-x / 2, -y / 2, -z / 2],
        max_pt=[x / 2, y / 2, z / 2],
        shape=[x, y, z],
        dtype="float32",
    )
    geometry = helical_geometry(config, reco_space)

    ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl="astra_cuda")
    fbp = odl.tomo.fbp_op(ray_trafo, filter_type="Hann", frequency_scaling=0.8)
    windowed_fbp = fbp * odl.tomo.tam_danielson_window(ray_trafo)
    return OperatorModule(ray_trafo), OperatorModule(windowed_fbp)


@torch.no_grad()
def simulate(config: Config, image: torch.Tensor):
    ray_trafo, fbp = get_operator(config, image)

    photons_per_pixel = config.data.photons_per_pixel
    sinogram = ray_trafo(image)

    normalized = sinogram / torch.amax(sinogram)
    noisy_p = torch.distributions.Poisson(photons_per_pixel * normalized)
    noisy = noisy_p.sample() / photons_per_pixel
    noisy = noisy * torch.amax(sinogram)
    return noisy, fbp(noisy)


def main(args):
    config_name = args.config
    atlas = args.atlas
    dist = args.dist
    workdir = args.workdir

    config = get_config(config_name)
    assert config.data.task == "reconstruction", "Only reconstruction task is supported"

    if atlas:
        results_dir = os.path.join(
            workdir, "results", "AbdomenAtlas", config.results_name
        )
        dataset = AbdomenAtlas(config, for_training=False, workdir=workdir)
    else:
        results_dir = config.get_results_dir(workdir=workdir)
        dataset = get_dataset(config, workdir=workdir)

    fbp_dir = os.path.join(results_dir, "fbp")
    os.makedirs(fbp_dir, exist_ok=True)

    rank, sampler = 0, None
    if dist:
        distributed.init_process_group(backend="nccl")
        rank = distributed.get_rank()
        sampler = DistributedSampler(dataset, even_divisible=False, shuffle=False)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=list_data_collate,
        sampler=sampler,
        num_workers=2,
    )

    for _, data in enumerate(tqdm(dataloader)):
        scan_name = data["name_img"][0]
        image = data["image"]

        image = image.to(device)

        _, reconstruction = simulate(config, image)
        reconstruction = reconstruction.squeeze()
        reconstruction = reconstruction.cpu().numpy()

        out_path = os.path.join(fbp_dir, f"{scan_name}.npy")
        np.save(out_path, reconstruction)

    if dist:
        distributed.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    main(args)
