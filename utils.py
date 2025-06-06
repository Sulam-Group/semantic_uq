from typing import Mapping, Optional

import numpy as np
import wandb


class Monitor:
    def __init__(self):
        self.data: Mapping[str, float] = {}
        self.global_samples: int = 0

    def zero(self):
        self.data = {}

    def update(
        self,
        data: Mapping[str, float],
        num_samples: int,
        increase_global_samples: bool = True,
    ):
        if increase_global_samples:
            self.global_samples += num_samples

        for k, v in data.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append((v, num_samples))

    def log(self, prefix: str, step: Optional[int] = None):
        data = {k: np.array(v) for k, v in self.data.items()}
        data = {k: np.sum(v[:, 0]) / sum(v[:, 1]) for k, v in data.items()}
        wandb.log(
            {f"{prefix}/{k}": v for k, v in data.items()},
            step=step or self.global_samples,
        )
        self.data = {}


organ_name_low = [
    "spleen",
    "kidney_right",
    "kidney_left",
    "gallbladder",
    "esophagus",
    "liver",
    "stomach",
    "aorta",
    "inferior_vena_cava",
    "portal_vein_and_splenic_vein",
    "pancreas",
    "adrenal_gland_right",
    "adrenal_gland_left",
    "duodenum",
    "hepatic_vessel",
    "lung_right",
    "lung_left",
    "colon",
    "intestine",
    "rectum",
    "bladder",
    "prostate",
    "femur_left",
    "femur_right",
    "celiac_truck",
    "kidney_tumor",
    "liver_tumor",
    "pancreas_tumor",
    "hepatic_vessel_tumor",
    "lung_tumor",
    "colon_tumor",
    "kidney_cyst",
]

target_template = [1, 2, 3, 4, 6, 7, 8, 9, 11]
organ_idx = [0, 1] + [idx + 1 for idx in target_template]
organ_names = ["background", "body"] + [
    organ_name_low[i - 1].replace("_", " ") for i in target_template
]
