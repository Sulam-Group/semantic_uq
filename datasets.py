import os
from abc import abstractmethod
from copy import deepcopy
from typing import Iterable, Mapping, Tuple

import nibabel as nib
import numpy as np
from monai.data import Dataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandAffined,
    RandSpatialCropd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)

from configs import Config
from configs import Constants as C
from utils import organ_name_low, target_template


class LoadFBPReconstruction(MapTransform):
    def __init__(self, fbp_dir: str, enabled: bool = True):
        super().__init__(keys="fbp")
        self.fbp_dir = fbp_dir
        self.enabled = enabled

    def __call__(self, data):
        if not self.enabled:
            return data
        scan_name = data["name_img"]
        reconstruction_path = os.path.join(self.fbp_dir, f"{scan_name}.npy")
        reconstruction = np.load(reconstruction_path)
        reconstruction = reconstruction[None, ...]
        data["fbp"] = reconstruction
        return data


def base_transform(
    keys: Iterable[str],
    pixdim: Tuple[float, float, float] = None,
    with_fbp: bool = False,
    fbp_dir: str = None,
):
    if with_fbp:
        assert fbp_dir is not None

    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(
                keys=keys, pixdim=pixdim, mode="bilinear", allow_missing_keys=True
            ),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0, b_max=1, clip=True
            ),
            CropForegroundd(keys=keys, source_key="image", allow_smaller=True),
            LoadFBPReconstruction(fbp_dir, enabled=with_fbp),
        ]
    )


class ExperimentDataset(Dataset):
    image_keys = ["image"]
    prediction_keys = ["q_lo", "mmse", "q_hi", "body"]
    seg_keys = ["combined_labels"]

    def __init__(
        self,
        config: Config,
        with_fbp: bool = False,
        with_prediction_results=False,
        with_segmentation_results=False,
        workdir=C.workdir,
    ):
        self.data_dir = self.get_data_dir(config, workdir=workdir)

        results_dir = config.get_results_dir(workdir=workdir)
        fbp_dir = os.path.join(results_dir, "fbp")
        prediction_dir = os.path.join(results_dir, "prediction")
        segmentation_dir = os.path.join(
            results_dir, "segmentation", config.data.suprem_backbone
        )

        keys = deepcopy(ExperimentDataset.image_keys)
        data_dicts = self.get_data_dicts()

        if with_prediction_results:
            keys += ExperimentDataset.prediction_keys

            denoised_scan_names = os.listdir(prediction_dir)
            data_dicts = [d for d in data_dicts if d["name_img"] in denoised_scan_names]
            for d in data_dicts:
                for key in ExperimentDataset.prediction_keys:
                    d[key] = os.path.join(
                        prediction_dir, d["name_img"], f"{key}.nii.gz"
                    )
        if with_segmentation_results:
            keys += ExperimentDataset.seg_keys

            seg_scan_names = os.listdir(segmentation_dir)
            data_dicts = [d for d in data_dicts if d["name_img"] in seg_scan_names]
            for d in data_dicts:
                for key in ExperimentDataset.seg_keys:
                    d[key] = os.path.join(
                        segmentation_dir, d["name_img"], f"{key}.nii.gz"
                    )

        transform = base_transform(
            keys,
            pixdim=config.data.pixdim,
            with_fbp=with_fbp,
            fbp_dir=fbp_dir,
        )
        if with_fbp:
            keys += ["fbp"]
        transform = Compose([transform, ToTensord(keys=keys)])
        super().__init__(data=data_dicts, transform=transform)

    @abstractmethod
    def get_data_dir(self, config: Config, workdir=C.workdir):
        pass

    @abstractmethod
    def get_data_dicts(self):
        data_dir = self.data_dir
        pass

    @abstractmethod
    def get_labels(self, idx):
        data_dir = self.data_dir
        pass


datasets: Mapping[str, ExperimentDataset] = {}


def register_dataset(name: str):
    def register(cls: ExperimentDataset):
        datasets[name] = cls
        return cls

    return register


def get_dataset(
    config: Config,
    with_fbp=False,
    with_prediction_results=False,
    with_segmentation_results=False,
    workdir=C.workdir,
) -> ExperimentDataset:
    Dataset = datasets[config.data.dataset]
    return Dataset(
        config,
        with_fbp=with_fbp,
        with_prediction_results=with_prediction_results,
        with_segmentation_results=with_segmentation_results,
        workdir=workdir,
    )


@register_dataset(name="TotalSegmentator")
class TotalSegmentator(ExperimentDataset):
    def __init__(
        self,
        config: Config,
        with_fbp: bool = False,
        with_prediction_results: bool = False,
        with_segmentation_results: bool = False,
        workdir=C.workdir,
    ):
        super().__init__(
            config,
            with_fbp=with_fbp,
            with_prediction_results=with_prediction_results,
            with_segmentation_results=with_segmentation_results,
            workdir=workdir,
        )

    def get_data_dir(self, config, workdir=C.workdir):
        return os.path.join(workdir, "data", config.data.dataset)

    def get_data_dicts(self):
        data_dir = self.data_dir

        scan_names = sorted(
            list(filter(lambda x: x.startswith("s"), os.listdir(data_dir)))
        )
        scan_paths = [
            os.path.join(data_dir, scan_name, "ct.nii.gz") for scan_name in scan_names
        ]
        return [
            {"image": path, "name_img": name, "path_img": path}
            for name, path in zip(scan_names, scan_paths)
        ]

    def get_labels(self, idx):
        data_dir = self.data_dir

        scan_name = self.data[idx]["name_img"]
        label_dir = os.path.join(data_dir, scan_name, "segmentations")

        labels = None
        for organ_target in target_template:
            organ_name = organ_name_low[organ_target - 1]
            organ_label_path = os.path.join(label_dir, f"{organ_name}.nii.gz")
            organ_label = nib.load(organ_label_path).get_fdata().astype(np.int64)
            if labels is None:
                labels = np.zeros_like(organ_label)
            labels[organ_label == 1] = organ_target
        return labels


@register_dataset(name="FLARE")
class FLARE(ExperimentDataset):
    def __init__(
        self,
        config: Config,
        with_fbp: bool = False,
        with_prediction_results: bool = False,
        with_segmentation_results: bool = False,
        workdir=C.workdir,
    ):
        super().__init__(
            config,
            with_fbp=with_fbp,
            with_prediction_results=with_prediction_results,
            with_segmentation_results=with_segmentation_results,
            workdir=workdir,
        )
        self.pixdim = config.data.pixdim

    def get_data_dir(self, config: Config, workdir=C.workdir):
        return os.path.join(
            workdir,
            "data",
            f"{config.data.dataset}_{','.join(map(str, config.data.pixdim))}",
        )

    def get_data_dicts(self):
        data_dir = self.data_dir

        with open(os.path.join(data_dir, "bad_scans.txt"), "r") as f:
            bad_scans = f.read().splitlines()

        scans = sorted(
            list(
                filter(
                    lambda x: x.endswith(".nii.gz") and (x not in bad_scans),
                    os.listdir(os.path.join(data_dir, "scans")),
                )
            )
        )
        scan_names = [s.split(".")[0] for s in scans]
        scan_paths = [os.path.join(data_dir, "scans", scan) for scan in scans]
        return [
            {"image": path, "name_img": name, "path_img": path}
            for name, path in zip(scan_names, scan_paths)
        ]

    def get_labels(self, idx):
        organ_map = {
            "liver": 1,
            "kidney_right": 2,
            "spleen": 3,
            "pancreas": 4,
            "aorta": 5,
            "inferior_vena_cava": 6,
            "right_adrenal_gland": 7,
            "left_adrenal_gland": 8,
            "gallbladder": 9,
            "esophagus": 10,
            "stomach": 11,
            "duodenum": 12,
            "kidney_left": 13,
            "tumor": 14,
        }
        data_dir = self.data_dir

        scan_name = self.data[idx]["name_img"]
        labels_path = os.path.join(
            data_dir, "labels", f"{scan_name.replace('_0000', '')}.nii.gz"
        )

        transform = Compose(
            [
                LoadImaged(keys=["labels"]),
                EnsureChannelFirstd(keys=["labels"]),
                Orientationd(keys=["labels"], axcodes="RAS"),
                Spacingd(keys=["labels"], pixdim=self.pixdim, mode="nearest"),
            ]
        )
        data = {"labels": labels_path}
        data = transform(data)
        orig_labels = data["labels"].squeeze()

        labels = np.zeros_like(orig_labels)
        for organ_target in target_template:
            organ_name = organ_name_low[organ_target - 1]
            orig_organ_target = organ_map[organ_name]
            labels[orig_labels == orig_organ_target] = organ_target
        return labels


class AbdomenAtlas(Dataset):
    def __init__(
        self,
        config: Config,
        for_training: bool = True,
        with_fbp: bool = False,
        workdir=C.workdir,
    ):
        data_dir = os.path.join(workdir, "data", "AbdomenAtlas")
        scan_names = sorted(
            list(filter(lambda x: x.startswith("BDMAP"), os.listdir(data_dir)))
        )
        data_dicts = [
            {
                "image": os.path.join(data_dir, scan_name, "ct.nii.gz"),
                "name_img": scan_name,
            }
            for scan_name in scan_names
        ]

        roi_size = config.data.roi_size
        keys = ["image"]
        transform = Compose(
            [
                base_transform(
                    keys,
                    pixdim=(1.5, 1.5, 1.5),
                    with_fbp=with_fbp,
                    fbp_dir=os.path.join(
                        workdir, "results", "AbdomenAtlas", config.results_name, "fbp"
                    ),
                ),
                SpatialPadd(keys, roi_size, mode="constant"),
            ]
        )

        if with_fbp:
            keys += ["fbp"]

        if for_training:
            transform = Compose(
                [
                    transform,
                    RandSpatialCropd(keys=keys, roi_size=roi_size, random_size=False),
                    RandAffined(
                        keys=keys,
                        mode="bilinear",
                        prob=1.0,
                        spatial_size=roi_size,
                        rotate_range=(0, 0, np.pi / 15),
                        scale_range=(0.3, 0.3, 0.3),
                    ),
                ]
            )
        transform = Compose([transform, ToTensord(keys=keys)])
        super().__init__(data=data_dicts, transform=transform)
