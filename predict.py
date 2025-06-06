import argparse
import os

import nibabel as nib
import numpy as np
import torch
import torch.distributed as distributed
from monai.data import (
    DataLoader,
    DistributedSampler,
    MetaTensor,
    decollate_batch,
    list_data_collate,
)
from monai.transforms import Invertd
from scipy.ndimage import binary_fill_holes
from skimage.measure import label
from skimage.morphology import binary_opening, dilation, disk
from tqdm import tqdm

from configs import Constants as C
from configs import get_config
from datasets import get_dataset
from model import Denoiser


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ts")
    parser.add_argument("--dist", action="store_true", default=False)
    parser.add_argument("--workdir", type=str, default=C.workdir)
    return parser.parse_args()


def get_body_mask(mse: torch.Tensor, body_threshold: float = 0.1):
    body_mask = (mse > body_threshold).squeeze().cpu().numpy()
    for slice_idx in range(body_mask.shape[-1]):
        slice_mask = body_mask[..., slice_idx]

        slice_mask = binary_opening(slice_mask, disk(3))
        slice_cc = label(slice_mask)
        nbins = np.bincount(slice_cc.flat)
        if len(nbins) == 1:
            slice_mask = np.zeros_like(slice_mask)
        else:
            largest_cc = np.argmax(nbins[1:]) + 1
            slice_mask = slice_cc == largest_cc
            slice_mask = dilation(slice_mask, disk(5))
            slice_mask = binary_fill_holes(slice_mask)

        body_mask[..., slice_idx] = slice_mask
    return torch.from_numpy(body_mask[None, None, ...]).to(mse.device)


@torch.no_grad()
def main(args):
    config_name = args.config
    dist = args.dist
    workdir = args.workdir

    config = get_config(config_name)
    prediction_dir = os.path.join(
        workdir, "results", config.data.dataset, config.results_name, "prediction"
    )
    os.makedirs(prediction_dir, exist_ok=True)

    prediction_keys = ["q_lo", "mmse", "q_hi", "body"]

    with_fbp = config.data.task == "reconstruction"
    dataset = get_dataset(config, with_fbp=with_fbp, workdir=workdir)
    post_transform = Invertd(
        keys=prediction_keys,
        transform=dataset.transform,
        orig_keys="image",
        meta_keys=[f"{key}_meta_dict" for key in prediction_keys],
        meta_key_postfix="meta_dict",
        nearest_interp=True,
        to_tensor=True,
    )

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
        num_workers=4,
    )

    model = Denoiser.from_pretrained(config, workdir=workdir, device=device)
    model.eval()

    for _, data in enumerate(tqdm(dataloader)):
        scan_name = data["name_img"][0]
        image_file_path = data["path_img"][0]

        original_affine = nib.load(image_file_path).affine
        original_data = nib.load(image_file_path).get_fdata()

        if config.data.task == "denoising":
            image = data["image"]
            image = image.to(device)
            measurement = image + config.data.sigma * torch.randn_like(image)
        elif config.data.task == "reconstruction":
            measurement = data["fbp"].to(device)

        output = model.denoise(measurement)
        q_lo = output[:, [0]]
        mmse = output[:, [1]]
        q_hi = output[:, [2]]

        body = get_body_mask(mmse)

        data["q_lo"] = q_lo
        data["mmse"] = mmse
        data["q_hi"] = q_hi
        data["body"] = MetaTensor(body.cpu().to(torch.uint8), meta=data["mmse"].meta)

        post_data = [post_transform(i) for i in decollate_batch(data)][0]

        out_dir = os.path.join(prediction_dir, scan_name)
        os.makedirs(out_dir, exist_ok=True)
        for key in prediction_keys:
            key_path = os.path.join(out_dir, f"{key}.nii.gz")
            key_data = post_data[key].squeeze(0).cpu().numpy()
            if key == "body":
                key_data = key_data.astype(np.uint8)
            else:
                key_data = np.clip(key_data, 0, 1)
            key_save = nib.Nifti1Image(key_data, original_affine)
            nib.save(key_save, key_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
