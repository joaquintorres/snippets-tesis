from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Union, Literal, Sequence, Dict, Tuple, Optional

import numpy as np

Color = Literal["r", "g", "b", "rgb"]


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

    @classmethod
    def from_json(cls, json_string: str) -> FpmConfig:
        data = json.loads(json_string)
        return cls(**data["fpm_config"])

    def to_dict(self) -> dict:
        data = {
            "wavelengths": self.wavelengths,
            "sample_height_mm": self.sample_height_mm,
            "center": self.center,
            "matrix_size": self.matrix_size,
            "led_gap_mm": self.led_gap_mm,
            "objetive_na": self.objetive_na,
            "pixel_size_um": self.pixel_size_um,
            "image_size": self.image_size,
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
        return f"{self.x:02d}_" f"{self.y:02d}_" f"{self.color}_" f"{self.exposition:d}"

    def get_name(self) -> str:
        return self.__str__() + ".npy"

    @classmethod
    def from_file(
        cls, path: Union[str, pathlib.Path], mmap: Optional[str] = "r"
    ) -> FpmImage:
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

        with open(path.joinpath("config.json"), mode="r") as f:
            config = FpmConfig.from_json(f.read())

        images = [
            FpmImage.from_file(file, mmap=mmap) for file in path.glob("??_??_*_*.npy")
        ]

        colors = list(set([image.color for image in images]))

        ordered_images = {
            color: {
                (image.x, image.y): image for image in images if image.color == color
            }
            for color in colors
        }

        return cls(path, colors, config, ordered_images)


@dataclass
class FpmAcquisitionSetup:
    name: Optional[str] = None
    description: Optional[str] = None
    sample_dir: Optional[pathlib.Path] = None
    size: Optional[int] = None
    colors: Optional[Sequence[Color]] = None
    red_exp: Optional[int] = None
    green_exp: Optional[int] = None
    blue_exp: Optional[int] = None
    scheme: Optional[str] = None
    min_exposure: Optional[int] = None
    max_exposure: Optional[int] = None
    config: Optional[pathlib.Path] = None
    exposure_file: Optional[pathlib.Path] = None
    bit_depth: Optional[int] = None
    dry_run: Optional[bool] = None
    dummy_devices: Optional[bool] = None
