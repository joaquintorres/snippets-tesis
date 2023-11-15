from typing import Sequence, Optional, Tuple, Union, Iterator

import numpy as np

from gigapixel.drivers import (
    QhyCamera,
    IlluminationDriver,
    Camera,
    Illumination,
    DummyIlluminationDriver,
    DummyCamera,
    Roi,
)

from gigapixel.drivers.imperx import ImperxCamera

from .interfaces import FpmImage, Color, FpmAcquisitionSetup, FpmStatus
from .tools import xy_iterator, get_scheme, f_scheme


def set_camera(dummy: bool, bit_depth: int, roi: Roi) -> Camera:
    if dummy:
        return DummyCamera(bit_depth=bit_depth, roi=roi)
    else:
        return ImperxCamera(bit_depth=bit_depth, roi=roi)


def set_illumination(dummy: bool, port: str) -> Illumination:
    if dummy:
        return DummyIlluminationDriver(port=port)
    else:
        print(f"OPEN {port}")
        return IlluminationDriver(port=port)


def limit_exposure(exp: Union[float, int], limits: Tuple[int, int]) -> int:
    return int(min(max(exp, limits[0]), limits[1]))


def acquire_single(
    camera: Camera,
    illumination: Illumination,
    color: Color = 'rgb',
    x: int = 15,
    y: int = 15,
    exposure: int = 1000
) -> FpmImage:
    illumination.turn_on_led(y, x, color=color)
    camera.set_gain_exposure(gain=100, exposure=exposure)
    frame = camera.get_frame()
    illumination.turn_off_leds()
    return FpmImage(frame, x, y, color, exposure)


def acquire_sequence(
    camera: Camera,
    illumination: Illumination,
    setup: FpmAcquisitionSetup,
    status: FpmStatus
) -> Iterator[FpmImage]:
    try:
        camera.set_gain_exposure(gain=100, exposure=5000)
        camera.get_frame()
        main_exposure = setup.get_exposures()
        iterator = xy_iterator(
            size=setup.size,
            center=setup.fpm_config.center,
            scheme=get_scheme(setup.scheme)
        )
        limits = (setup.min_exposure, setup.max_exposure)

        for x, y in iterator:
            for i, color in enumerate(setup.colors):
                _exp = main_exposure[i] * setup.exposure_matrix[x, y]
                exp = limit_exposure(_exp, limits)
                yield acquire_single(
                    camera=camera,
                    illumination=illumination,
                    color=color,
                    x=x,
                    y=y,
                    exposure=exp
                )
                status.add(x, y, color)
    finally:
        illumination.turn_off_leds()
