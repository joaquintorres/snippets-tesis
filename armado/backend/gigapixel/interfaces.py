from __future__ import annotations

import json
import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Union, Literal, Sequence, Dict, Tuple, Optional

import numpy as np

from .tools import xy_iterator, Scheme, get_scheme

Color = Literal["r", "g", "b", "rgb"]


def default_sample_path() -> pathlib.Path:
    path = pathlib.Path(os.environ.get("HOME", "")).absolute() / "fpm_samples"
    path.mkdir(exist_ok=True, parents=True)
    print(path)
    return path


def default_config_path() -> pathlib.Path:
    base_path = pathlib.Path(__file__).parents[1]
    return base_path / "config" / "default.json"


def default_exposure(center: Tuple[int, int]) -> np.ndarray:
    xx, yy = np.meshgrid(np.arange(32), np.arange(32))
    sigma = 4.5847
    xx -= center[0]
    yy -= center[1]
    median = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return 1 / median


@dataclass
class FpmConfig:
    wavelengths: Dict[Color, float]
    sample_height_mm: float
    center: Tuple[int, int]
    matrix_size: int
    led_gap_mm: float
    objetive_na: float
    pixel_size_um: float
    image_size: Tuple[int, int]
    camera_min_time: float

    @classmethod
    def from_json(cls, json_string: str) -> FpmConfig:
        data = json.loads(json_string)
        return cls(**data["fpm_config"])

    @classmethod
    def make_default(cls) -> FpmConfig:
        with open(default_config_path()) as f:
            default = cls.from_json(f.read())
        return default

    def to_dict(self) -> dict:
        data = {
            "wavelengths": self.wavelengths,
            "sample_height_mm": self.sample_height_mm,
            "center": self.center,
            "matrix_size": self.matrix_size,
            "led_gap_mm": self.led_gap_mm,
            "objetive_na": self.objetive_na,
            "pixel_size_um": self.pixel_size_um,
            "image_size": self.image_size
        }
        return data


@dataclass
class FpmImage:
    frame: np.ndarray
    x: int
    y: int
    color: Color
    exposition: Union[int, float]

    def __str__(self):
        return (
            f"{self.x:02d}_"
            f"{self.y:02d}_"
            f"{self.color}_"
            f"{self.exposition:d}"
        )

    def get_name(self) -> str:
        return self.__str__() + ".npy"

    @classmethod
    def from_file(cls, path: Union[str, pathlib.Path],
                  mmap: Optional[str] = "r") -> FpmImage:
        if isinstance(path, str):
            path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        if not path.is_file():
            raise IsADirectoryError(str(path))

        parts = path.stem.split("_")
        x = int(parts[0])
        y = int(parts[1])
        color = parts[2]
        exposition = int(parts[3])
        frame = np.load(str(path), mmap_mode=mmap)
        return cls(frame, x, y, color, exposition)

    def save(self, path: Union[str, pathlib.Path]) -> str:
        name = path.joinpath(self.get_name())
        np.save(name, self.frame)
        return name


@dataclass
class FpmDataset:
    path: pathlib.Path
    colors: Sequence[Color]
    config: FpmConfig
    images: Dict[Color, Dict[Tuple[int, int], FpmImage]]

    @classmethod
    def from_path(
            cls, path: Union[str, pathlib.Path], mmap: Optional[str] = "r"
    ) -> FpmDataset:
        if isinstance(path, str):
            path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        if not path.is_dir():
            raise NotADirectoryError(str(path))

        with open(path.joinpath("config.json"), mode='r') as f:
            config = FpmConfig.from_json(f.read())

        images = [
            FpmImage.from_file(file, mmap=mmap)
            for file in path.glob("??_??_*_*.npy")
        ]

        colors = list(set([
            image.color
            for image in images
        ]))

        ordered_images = {
            color: {
                (image.x, image.y): image
                for image in images
                if image.color == color
            }
            for color in colors
        }

        return cls(path, colors, config, ordered_images)


@dataclass
class FpmAcquisitionSetup:
    name: str
    description: str
    sample_dir: pathlib.Path
    size: int
    colors: Sequence[Color]
    red_exp: int
    green_exp: int
    blue_exp: int
    scheme: str
    min_exposure: int
    max_exposure: int
    fpm_config: FpmConfig
    exposure_matrix: np.ndarray
    bit_depth: int
    dry_run: bool
    dummy_devices: bool

    @classmethod
    def make_default(cls):
        fpm_config = FpmConfig.make_default()
        return cls(
            name="",
            description="",
            sample_dir=default_sample_path(),
            size=16,
            colors='rgb',
            red_exp=2000,
            green_exp=2000,
            blue_exp=2000,
            scheme="square",
            min_exposure=50,
            max_exposure=1000000,
            fpm_config=fpm_config,
            exposure_matrix=default_exposure(fpm_config.center),
            bit_depth=16,
            dry_run=False,
            dummy_devices=False,
        )

    def get_exposures(self) -> Sequence[int]:
        all_exposures = [self.red_exp, self.green_exp, self.blue_exp]
        return [
            value
            for color, value in zip("rgb", all_exposures)
            if color in self.colors
        ]

    def get_scheme_function(self) -> Scheme:
        return get_scheme(self.scheme)

    def get_scheme_sequence(self):
        iterator = xy_iterator(
            size=self.size,
            center=self.fpm_config.center,
            scheme=self.get_scheme_function()
        )
        return np.array([
            [x, y]
            for x, y in iterator
        ])

    def get_time_matrix(self) -> np.ndarray:
        used_exposures = np.zeros_like(self.exposure_matrix)
        seq = self.get_scheme_sequence()
        nc = len(self.colors)
        for x, y in seq:
            used_exposures[y, x] = self.exposure_matrix[y, x]
        exposures = np.array([
            np.clip(used_exposures * exp, self.min_exposure, self.max_exposure)
            for exp in self.get_exposures()
        ])
        exposures /= 1e6

        for x, y in seq:
            exposures[:, y, x] += self.fpm_config.camera_min_time * nc

        return exposures

    def get_estimated_time(self) -> float:
        return float(np.sum(self.get_time_matrix()))

    def get_image_size(self) -> int:
        return self.fpm_config.image_size[0] * self.fpm_config.image_size[1]

    def get_estimated_size(self) -> float:
        seq = self.get_scheme_sequence()
        file_size = (self.get_image_size() * (self.bit_depth / 8)) / 1.0e9
        print(seq.shape, self.size, self.bit_depth, file_size)
        estimated_size = file_size * seq.shape[0] * len(self.colors)
        return estimated_size


@dataclass
class FpmStatus:
    setup: FpmAcquisitionSetup
    running: bool = False
    time_matrix: np.ndarray = field(init=False)
    completed: np.ndarray = field(init=False)

    def __post_init__(self):
        self.recalculate()

    @property
    def progress(self) -> float:
        completed = np.sum(self.time_matrix[self.completed])
        total = np.sum(self.time_matrix)
        return completed / total

    @property
    def eta(self) -> float:
        completed = np.sum(self.time_matrix[self.completed])
        total = np.sum(self.time_matrix)
        return total - completed

    def recalculate(self):
        self.time_matrix = self.setup.get_time_matrix()
        self.completed = np.zeros_like(self.time_matrix, dtype=bool)

    def add(self, x: int, y: int, color: str):
        print(np.sum(self.completed))
        self.completed[["r", "g", "b"].index(color), x, y] = True
