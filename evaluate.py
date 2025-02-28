import argparse
import os

import numpy as np
import pandas as pd
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)
from skimage.metrics import structural_similarity as ssim_fn
from sklearn.metrics import f1_score
from tqdm import tqdm

from calibrate import load_abdomen_window
from configs import Constants as C
from configs import get_config
from datasets import get_dataset
from utils import organ_name_low, target_template

rng = np.random.default_rng()
target_template = np.array(target_template)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--workdir", type=str, default=C.workdir)
    return parser.parse_args()


def main(args):
    config_name = args.config
    n = args.n
    workdir = args.workdir

    config = get_config(config_name)
    dataset = get_dataset(
        config,
        with_prediction_results=True,
        with_segmentation_results=True,
        workdir=workdir,
    )
    abdomen_window = load_abdomen_window(config, dataset, workdir=workdir)
    abdomen_window.set_index("scan_name", inplace=True)

    keys = ["image", "mmse", "combined_labels"]
    dataset.transform = Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(
                keys=keys,
                pixdim=config.data.pixdim,
                mode="bilinear",
                allow_missing_keys=True,
            ),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0, b_max=1, clip=True
            ),
        ]
    )

    scan_names = []
    psnr = np.zeros(n)
    ssim = np.zeros(n)
    f1 = np.zeros((n, len(target_template)))

    idx = rng.permutation(abdomen_window["idx"].values)
    pbar = tqdm(total=n)
    pbar.update(0)

    i = 0
    for _idx in idx:
        data = dataset[_idx]
        scan_name = data["name_img"]

        image = data["image"].squeeze()
        mmse = data["mmse"].squeeze()
        segmentation = data["combined_labels"].squeeze()
        labels = dataset.get_labels(_idx)
        assert image.shape == mmse.shape == segmentation.shape == labels.shape

        unique_labels = np.unique(labels)
        if len(unique_labels) == 1 and unique_labels[0] == 0:
            continue
        if (
            config.data.dataset == "FLARE"
            and len(unique_labels) == 2
            and unique_labels[-1] == 14
        ):
            continue

        scan_names.append(scan_name)

        window = abdomen_window.loc[scan_name, "window_slice_idx"]
        image = image[..., window]
        mmse = mmse[..., window]
        segmentation = segmentation[..., window]
        labels = labels[..., window]

        mse = (image - mmse) ** 2
        mse = np.mean(mse)
        psnr[i] = 10 * np.log10(1 / mse)

        for j in range(len(window)):
            _image = image[..., j]
            _mmse = mmse[..., j]
            ssim[i] += ssim_fn(_image, _mmse, data_range=1.0)
        ssim[i] /= len(window)

        segmentation = segmentation.flatten()
        labels = labels.flatten()

        target_mask = np.isin(target_template, np.unique(labels), assume_unique=True)
        f1[i][target_mask] = f1_score(
            labels,
            segmentation,
            labels=target_template[target_mask],
            average=None,
            zero_division=np.nan,
        )
        f1[i][~target_mask] = np.nan

        i += 1
        pbar.update(1)
        if i == n:
            break

    data = {"scan_name": scan_names, "psnr": psnr, "ssim": ssim}
    for j, organ_target in enumerate(target_template):
        organ_name = organ_name_low[organ_target - 1]
        data[organ_name] = f1[:, j]

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(config.get_results_dir(workdir=workdir), "evaluation.csv"))


if __name__ == "__main__":
    args = parse_args()
    main(args)
