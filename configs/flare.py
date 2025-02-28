import numpy as np

from configs.utils import Config, register_config


@register_config(name="flare")
class FLARE(Config):
    def __init__(self):
        super().__init__()
        self.data.dataset = "FLARE"
        self.data.pixdim = (1.5, 1.5, 3.0)
        self.data.roi_size = (96, 96, 96)
        self.data.q_lo = 0.1
        self.data.q_hi = 0.9
        self.data.suprem_backbone = "unet"

        self.data.task = ["denoising", "reconstruction"]
        # denoising config
        self.data.sigma = 0.2
        # reconstruction config
        self.data.src_radius = 1000
        self.data.det_radius = 500
        self.data.num_turns = 8
        self.data.photons_per_pixel = 1000
        self.data.max_det_shape = [512, 128]
        self.data.max_num_angles = 1000

        self.calibration.r = 10
        self.calibration.image_size = 256
        self.calibration.window_size = 48
        self.calibration.window_slices = 4

        self.calibration.procedure = [
            # "rcps",
            # "krcps",
            "semrcps",
        ]
        self.calibration.uq = "qr_additive"
        self.calibration.loss = "01"
        self.calibration.bound = "crc"

        self.calibration.epsilon = 0.1
        self.calibration.delta = 0.05
        self.calibration.n_cal = 512
        self.calibration.n_val = 128

        self.calibration.lambda_max = 0.15
        self.calibration.stepsize = 2e-03

        self.calibration.n_opt = 32
        self.calibration.gamma = np.linspace(0.3, 0.7, 16)

        # k-RCPS config
        self.calibration.membership = None  # set automatically depending on uq
        self.calibration.k = 4
        self.calibration.prob_size = 50

        # seg-RCPS config
        self.calibration.min_support = 2
        self.calibration.max_support = 1000
        self.calibration.sem_control = [True]
