import os
import pickle
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, Mapping, Optional, Tuple

import torch
from krcps import Config as _CalibrationConfig
from ml_collections import ConfigDict


@dataclass
class Constants:
    workdir: str = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get(dict, key):
    subkeys = key.split(".")
    if len(subkeys) == 1:
        return dict[subkeys[0]]
    else:
        return _get(dict[subkeys[0]], ".".join(subkeys[1:]))


def _set(dict, key, value):
    subkeys = key.split(".")
    if len(subkeys) == 1:
        dict[subkeys[0]] = value
    else:
        _set(dict[subkeys[0]], ".".join(subkeys[1:]), value)


class DataConfig(ConfigDict):
    def __init__(self, config_dict: Mapping[str, Any] = {}):
        super().__init__()
        self.dataset: str = config_dict.get("dataset", None)
        self.pixdim: Tuple[float, float, float] = config_dict.get("pixdim", None)
        self.roi_size: Tuple[int, int, int] = config_dict.get("roi_size", None)
        self.q_lo: float = config_dict.get("q_lo", None)
        self.q_hi: float = config_dict.get("q_hi", None)
        self.suprem_backbone: str = config_dict.get("suprem_backbone", None)

        self.task: str = config_dict.get("task", None)

        # denoising config
        self.sigma: float = config_dict.get("sigma", None)

        # reconstruction config
        self.src_radius: float = config_dict.get("src_radius", None)
        self.det_radius: float = config_dict.get("det_radius", None)
        self.num_turns: float = config_dict.get("num_turns", None)
        self.max_det_shape: Tuple[int, int] = config_dict.get("max_det_shape", None)
        self.max_num_angles: float = config_dict.get("max_num_angles", None)
        self.photons_per_pixel: float = config_dict.get("photons_per_pixel", None)


class CalibrationConfig(_CalibrationConfig):
    def __init__(self, config_dict: Mapping[str, Any] = {}):
        super().__init__(config_dict=config_dict)
        self.r: int = config_dict.get("r", None)
        self.window_size: int = config_dict.get("window_size", None)
        self.window_slices: int = config_dict.get("window_slices", None)
        self.image_size: int = config_dict.get("image_size", None)


class Config(ConfigDict):
    def __init__(self, config_dict: Mapping[str, Any] = {}):
        super().__init__()

        self.data = DataConfig(config_dict.get("data", {}))
        self.calibration = CalibrationConfig(config_dict.get("calibration", {}))

    @property
    def results_name(self):
        if self.data.task == "denoising":
            results_name = f"denoising_sigma{self.data.sigma}"
        elif self.data.task == "reconstruction":
            results_name = (
                f"reconstruction_src{self.data.src_radius}"
                f"_det{self.data.det_radius}"
                f"_turns{self.data.num_turns}"
                f"_photons{self.data.photons_per_pixel}"
            )
        else:
            raise ValueError(f"Task {self.data.task} not supported")
        return results_name

    def get_results_dir(self, workdir=Constants.workdir):
        return os.path.join(workdir, "results", self.data.dataset, self.results_name)

    def get_calibration_results(
        self,
        results_kwargs: Mapping[str, Any] = {},
        workdir=Constants.workdir,
    ):
        config = self
        config_dict = self.to_dict()
        for key, value in results_kwargs.items():
            _set(config_dict, key, value)
        config = Config(config_dict=config_dict)
        return [CalibrationResults.load(c, workdir=workdir) for c in config.sweep()]

    def _sweep(self, keys=Iterable[str]):
        config_dict = self.to_dict()
        sweep_values = [_get(config_dict, key) for key in keys]
        sweep = list(
            product(*map(lambda x: x if isinstance(x, list) else [x], sweep_values))
        )

        configs: Iterable[Config] = []
        for _sweep in sweep:
            _config_dict = config_dict.copy()
            for key, value in zip(keys, _sweep):
                _set(_config_dict, key, value)
            _config = Config(config_dict=_config_dict)
            if _config.calibration.procedure == "krcps":
                if isinstance(_config.calibration.uq, str):
                    if _config.calibration.uq == "qr_additive":
                        _config.calibration.membership = "01_loss_quantile"
                    elif _config.calibration.uq == "mse":
                        _config.calibration.membership = "mse_loss_quantile"
                    elif _config.calibration.uq.startswith("seg"):
                        continue
                    else:
                        raise ValueError(
                            f"uq {_config.calibration.uq} not supported for krcps sweep"
                        )

            configs.append(_config)
        return configs

    def sweep(self):
        configs: Iterable[Config] = []

        procedure_configs = self._sweep(
            keys=[
                "data.task",
                "calibration.procedure",
                "calibration.uq",
                "calibration.bound",
            ]
        )
        for procedure_config in procedure_configs:
            if procedure_config.calibration.procedure == "rcps":
                configs.append(procedure_config)
            elif procedure_config.calibration.procedure == "krcps":
                procedure_keys = [
                    "calibration.n_opt",
                    "calibration.membership",
                    "calibration.k",
                    "calibration.prob_size",
                ]
                configs.extend(procedure_config._sweep(keys=procedure_keys))
            elif procedure_config.calibration.procedure == "semrcps":
                procedure_keys = [
                    "calibration.n_opt",
                    "calibration.min_support",
                    "calibration.sem_control",
                ]
                configs.extend(procedure_config._sweep(keys=procedure_keys))
            else:
                raise ValueError(
                    f"Procedure {procedure_config.calibration.procedure} not supported"
                    " for automatic sweep"
                )
        return configs


configs: Mapping[str, Config] = {}


def register_config(name: str):
    def register(cls):
        configs[name] = cls
        return cls

    return register


def get_config(name: str) -> Config:
    return configs[name]()


@dataclass
class CalibrationResult:
    cal_idx: Iterable[int] = None
    val_idx: Iterable[int] = None
    _lambda: torch.Tensor = None
    loss: float = None
    organ_loss: torch.Tensor = None
    i_mean: float = None
    organ_i_mean: torch.Tensor = None


class CalibrationResults:
    def __init__(self, config: Config):
        self._index = 0

        self.config = config
        self.data: list[CalibrationResult] = []

    def get(self, key: str):
        return [getattr(result, key) for result in self.data]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __getitem__(self, idx) -> CalibrationResult:
        return self.data[idx]

    def __next__(self) -> CalibrationResult:
        if self._index >= len(self.data):
            raise StopIteration
        result = self.data[self._index]
        self._index += 1
        return result

    @classmethod
    def load(cls, config: Config, workdir=Constants.workdir):
        calibration_dir = os.path.join(
            config.get_results_dir(workdir=workdir), "calibration"
        )

        results = cls(config)
        with open(os.path.join(calibration_dir, f"{results.name()}.pkl"), "rb") as f:
            data = pickle.load(f)

        results.data = data
        return results

    def append(self, **kwargs):
        self.data.append(CalibrationResult(**kwargs))

    def procedure_name(self):
        if self.config.calibration.procedure == "rcps":
            name = "CRC"
        elif self.config.calibration.procedure == "krcps":
            name = r"$K$-CRC"
        elif self.config.calibration.procedure == "semrcps":
            name = r"$sem$-CRC"
            if self.config.calibration.sem_control:
                name = r"$\overline{sem}$-CRC"

        # if self.config.calibration.uq == "qr_additive":
        #     name = f"{name}\n(QR"
        # elif self.config.calibration.uq == "mse":
        #     name = f"{name}\n(MSE"

        # if self.config.calibration.bound == "hoeffding_bentkus":
        #     name = f"{name}, HB)"
        # elif self.config.calibration.bound == "crc":
        #     name = f"{name}, CRC)"
        return name

    def name(self):
        procedure = self.config.calibration.procedure
        uq = self.config.calibration.uq
        bound = self.config.calibration.bound
        epsilon = self.config.calibration.epsilon
        delta = self.config.calibration.delta
        name = f"{procedure}_{uq}_{bound}_eps{epsilon}_delta{delta}"
        if procedure == "krcps":
            n_opt = self.config.calibration.n_opt
            k = self.config.calibration.k
            prob_size = self.config.calibration.prob_size
            name += f"_nopt{n_opt}_k{k}_size{prob_size}"
        if procedure == "semrcps":
            n_opt = self.config.calibration.n_opt
            min_support = self.config.calibration.min_support
            name += f"_nopt{n_opt}_minsup{min_support}"
            if self.config.calibration.sem_control:
                name += "_semcontrol"
        return name

    def save(self, workdir=Constants.workdir):
        calibration_dir = os.path.join(
            self.config.get_results_dir(workdir=workdir), "calibration"
        )
        os.makedirs(calibration_dir, exist_ok=True)

        with open(os.path.join(calibration_dir, f"{self.name()}.pkl"), "wb") as f:
            pickle.dump(self.data, f)
