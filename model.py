import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.networks.layers import Act, Norm
from monai.networks.nets import UNet

from configs import Config
from configs import Constants as C


class Denoiser(nn.Module):
    def __init__(self, config: Config, device=C.device):
        super().__init__()
        self.roi_size = config.data.roi_size

        self.unet = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(1, 1, 1, 1),
            num_res_units=2,
            act=Act.GELU,
            norm=Norm.BATCH,
        )

        self.to(device)

    @classmethod
    def from_pretrained(cls, config: Config, workdir=C.workdir, device=C.device):
        model = cls(config, device=device)
        weights_dir = os.path.join(workdir, "weights", config.results_name)
        with open(os.path.join(weights_dir, "latest.txt"), "r") as f:
            latest = f.read().strip()
        checkpoint = torch.load(os.path.join(weights_dir, latest), map_location=device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model

    def forward(self, measurement):
        return self.unet(measurement)

    @torch.no_grad()
    def denoise(self, measurement):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = sliding_window_inference(
                measurement, self.roi_size, 1, self, mode="gaussian"
            )
        return output
