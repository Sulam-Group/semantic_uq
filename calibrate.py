import argparse
import logging
import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from krcps import get_loss, get_procedure, get_uq
from monai.data import Dataset
from monai.transforms import Compose, MapTransform, ResizeWithPadOrCropd
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from configs import CalibrationResults, Config
from configs import Constants as C
from configs import get_config
from datasets import get_dataset

logger = logging.getLogger(__name__)


def setup_logging(level):
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.root.setLevel(level)
    loggers = [
        logging.getLogger(name)
        for name in logging.root.manager.loggerDict
        if "ibydmt" in name
    ]
    for logger in loggers:
        logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ts")
    parser.add_argument("--workdir", type=str, default=C.workdir)
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser.parse_args()


def get_abdomen_window(
    config: Config,
    segmentation: np.ndarray,
    label_threshold: float = 25e03,
    trim: int = 10,
) -> Optional[Iterable[int]]:
    window_size = config.calibration.window_size
    n_window_slices = config.calibration.window_slices

    if segmentation.shape[-1] < window_size + 2 * trim:
        return None

    n_labels = np.sum(segmentation, axis=(0, 1))
    n_labels = n_labels[trim:-trim]
    n_labels = n_labels * (n_labels >= label_threshold)
    if np.sum(n_labels) == 0:
        return None

    window_n_labels = np.convolve(n_labels, np.ones(window_size), mode="valid")
    start_slice_idx = np.argmax(window_n_labels)
    end_slice_idx = start_slice_idx + window_size
    if np.sum(n_labels[start_slice_idx:end_slice_idx] > 0) < window_size / 2:
        return None

    return (
        (
            np.linspace(start_slice_idx, end_slice_idx - 1, n_window_slices, dtype=int)
            + trim
        ).tolist(),
        n_labels,
    )


def load_abdomen_window(config: Config, dataset: Dataset, workdir=C.workdir):
    window_data_path = os.path.join(
        config.get_results_dir(workdir=workdir), "window.csv"
    )
    if not os.path.exists(window_data_path):
        logger.info("Creating window data...")

        def f(idx, data):
            scan_name = data["name_img"]
            data = dataset[idx]

            segmentation = data["combined_labels"].squeeze().numpy()
            scan_length = segmentation.shape[-1]

            window = get_abdomen_window(config, segmentation)
            if window is None:
                window = "no_abdomen_window"
            else:
                window = ",".join(map(str, window[0]))
            return idx, scan_name, scan_length, window

        window_data = Parallel(n_jobs=16)(
            delayed(f)(idx, data) for idx, data in enumerate(tqdm(dataset))
        )
        window_data = pd.DataFrame(
            window_data, columns=["idx", "scan_name", "scan_length", "window"]
        )
        window_data.to_csv(window_data_path, index=False)

    return pd.read_csv(window_data_path)


def gather_data(
    config: Config,
    idx: Optional[Iterable[int]] = None,
    n: Optional[int] = None,
    workdir=C.workdir,
):
    logger.info("Gathering data...")

    class AddBodyMask(MapTransform):
        def __init__(self, keys: Iterable[str]):
            super().__init__(keys)

        def __call__(self, data):
            d = dict(data)
            for key in self.key_iterator(d):
                d[key] = d[key] + d["body"]
            return d

    class TakeAbdomenWindowd(MapTransform):
        def __init__(self, keys: Iterable[str], abdomen_window: pd.DataFrame):
            super().__init__(keys)
            self.abdomen_window = abdomen_window.set_index("scan_name")

        def __call__(self, data):
            d = dict(data)
            scan_name = d["name_img"]
            scan_window = self.abdomen_window.loc[scan_name]
            window = scan_window["window"]
            window = list(map(int, window.split(",")))
            for key in self.key_iterator(d):
                try:
                    d[key] = d[key][..., window]
                except Exception as e:
                    print(e)
                    print(scan_name, d[key].shape, window)
            return d

    class EnsureZAxisFirstd(MapTransform):
        def __init__(self, keys: Iterable[str]):
            super().__init__(keys)

        def __call__(self, data):
            d = dict(data)
            for key in self.key_iterator(d):
                d[key] = d[key].permute(0, 3, 1, 2)
            return d

    dataset = get_dataset(
        config,
        with_prediction_results=True,
        with_segmentation_results=True,
        workdir=workdir,
    )

    abdomen_window = load_abdomen_window(config, dataset)

    if idx is None:
        valid = abdomen_window["window"] != "no_abdomen_window"
        valid_idx = abdomen_window[valid]["idx"].values
        perm_idx = np.random.permutation(valid_idx)
        if n is None:
            n = config.calibration.n_cal + config.calibration.n_val
        idx = perm_idx[:n]

    dataset = Subset(dataset, idx)

    window_slices = config.calibration.window_slices
    image_size = config.calibration.image_size
    ground_truth = torch.zeros(len(dataset), window_slices, image_size, image_size)
    reconstruction = torch.zeros(len(dataset), 3, window_slices, image_size, image_size)
    segmentation = torch.zeros(len(dataset), window_slices, image_size, image_size)

    keys = ["image", "q_lo", "mmse", "q_hi", "combined_labels"]
    dataset.dataset.transform = Compose(
        [
            dataset.dataset.transform,
            AddBodyMask(keys=["combined_labels"]),
            TakeAbdomenWindowd(keys, abdomen_window),
            ResizeWithPadOrCropd(keys, (image_size, image_size, window_slices)),
            EnsureZAxisFirstd(keys),
        ]
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    for i, data in enumerate(tqdm(dataloader)):
        ground_truth[i] = data["image"]
        reconstruction[i, 0] = data["q_lo"]
        reconstruction[i, 1] = data["mmse"]
        reconstruction[i, 2] = data["q_hi"]
        segmentation[i] = data["combined_labels"]
    return idx, ground_truth, reconstruction, segmentation


def calibrate(config: Config, workdir=C.workdir):
    message = (
        f"Calibrating dataset: {config.data.dataset} with\n"
        f"\tcalibration procedure: {config.calibration.procedure}\n"
        f"\tuncertainty quantification: {config.calibration.uq}\n"
        f"\tloss: {config.calibration.loss}\n"
        f"\tbound: {config.calibration.bound}\n"
        f"\tepsilon={config.calibration.epsilon}, delta={config.calibration.delta}\n"
        f"\tn_cal={config.calibration.n_cal}, n_val={config.calibration.n_val}\n"
    )
    if config.calibration.procedure == "krcps":
        message += (
            f"\tmembership={config.calibration.membership}\n\tn_opt={config.calibration.n_opt},"
            f" k={config.calibration.k}, prob_size={config.calibration.prob_size}"
        )
    if config.calibration.procedure == "semrcps":
        message += (
            f"\tn_opt={config.calibration.n_opt},"
            f" min_support={config.calibration.min_support},"
            f" max_support={config.calibration.max_support}\n"
            f"\tsem_control={config.calibration.sem_control}"
        )

    logger.info(message)

    calibration_procedure = get_procedure(config.calibration)

    idx, ground_truth, reconstruction, segmentation = gather_data(
        config, workdir=workdir
    )

    results = CalibrationResults(config)

    n = ground_truth.size(0)
    for t in range(config.calibration.r):
        logger.info(f"Running calibration iteration {t+1}/{config.calibration.r}")
        perm_idx = np.random.permutation(n)
        cal_idx = perm_idx[: config.calibration.n_cal]
        val_idx = perm_idx[config.calibration.n_cal :]

        cal_ground_truth, cal_reconstruction, cal_segmentation = (
            ground_truth[cal_idx],
            reconstruction[cal_idx],
            segmentation[cal_idx],
        )
        val_ground_truth, val_reconstruction, val_segmentation = (
            ground_truth[val_idx],
            reconstruction[val_idx],
            segmentation[val_idx],
        )

        uq = get_uq(config.calibration.uq)
        loss_fn = get_loss(config.calibration.loss)
        if config.calibration.procedure == "semrcps":
            _lambda = calibration_procedure(
                cal_ground_truth, cal_reconstruction, cal_segmentation
            )
            uq_fn = uq(val_reconstruction, val_segmentation)
        else:
            _lambda = calibration_procedure(cal_ground_truth, cal_reconstruction)
            uq_fn = uq(val_reconstruction)

        val_l, val_u = uq_fn(_lambda)
        val_i = val_u - val_l

        loss = loss_fn(val_ground_truth, val_l, val_u)

        organ_loss_fn = get_loss(
            "sem_01", segmentation=val_segmentation, target=val_ground_truth
        )
        organ_loss = organ_loss_fn(val_l, val_u)

        val_m = F.one_hot(val_segmentation.long()).float()
        norm = torch.sum(val_m.view(-1, val_m.size(-1)), dim=0)

        organ_i = val_m * val_i[..., None]
        organ_i = organ_i.view(-1, val_m.size(-1))
        organ_i_mean = torch.sum(organ_i, dim=0) / (norm + 1e-08)

        logger.info(
            f"validation loss: {loss.item():.2f}, mean interval length ="
            f" {val_i.mean().item():.4f}"
        )

        results.append(
            cal_idx=idx[cal_idx],
            val_idx=idx[val_idx],
            _lambda=_lambda,
            loss=loss.item(),
            organ_loss=organ_loss,
            i_mean=val_i.mean().item(),
            organ_i_mean=organ_i_mean,
        )
    # results.save(workdir=workdir)


def main(args):
    config_name = args.config
    workdir = args.workdir
    log_level = args.log_level

    setup_logging(log_level)

    config = get_config(config_name)

    for _config in config.sweep():
        calibrate(_config, workdir=workdir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
