import ctypes
import time

import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from typing import Tuple, Optional


Roi = Tuple[Tuple[int, int], Tuple[int, int]]


class Camera(ABC):
    @abstractmethod
    def __init__(self, bit_depth: int = 8, roi: Optional[Roi] = None):
        pass

    @abstractmethod
    def set_gain_exposure(self, gain: int, exposure: int):
        pass

    @abstractmethod
    def get_frame(self) -> np.ndarray:
        pass


class QhyCamera(Camera):
    def __init__(self, bit_depth: int = 8, roi: Optional[Roi] = None):
        self._qhyccd = ctypes.CDLL('libqhyccd.so')
        self._qhyccd.GetQHYCCDParam.restype = ctypes.c_double
        self._qhyccd.OpenQHYCCD.restype = ctypes.POINTER(ctypes.c_uint32)

        result = self._qhyccd.InitQHYCCDResource()
        if result == 0:
            print("InitSDK success\n")
        else:
            raise Exception('No SDK')

        cameras_found = self._qhyccd.ScanQHYCCD()

        if cameras_found > 0:
            print("found camera\n")
        else:
            raise Exception('No Camera')

        position_id = 0
        type_char_array_32 = ctypes.c_char * 32
        id_object = type_char_array_32()
        result = self._qhyccd.GetQHYCCDId(position_id, id_object)

        self._camera_handle = self._qhyccd.OpenQHYCCD(id_object)

        self._qhyccd.SetQHYCCDStreamMode(self._camera_handle, ctypes.c_uint32(0))
        self._qhyccd.InitQHYCCD(self._camera_handle)

        self._chipWidthMM = ctypes.c_uint32(0)
        self._chipHeightMM = ctypes.c_uint32(0)
        self._maxImageSizeX = ctypes.c_uint32(0)
        self._maxImageSizeY = ctypes.c_uint32(0)
        self._pixelWidthUM = ctypes.c_uint32(0)
        self._pixelHeightUM = ctypes.c_uint32(0)
        self._bpp = ctypes.c_uint32(0)

        self._depth = ctypes.c_uint32(bit_depth)
        self._channels = ctypes.c_uint32(1)

        self._qhyccd.GetQHYCCDChipInfo(
            self._camera_handle,
            ctypes.byref(self._chipWidthMM),
            ctypes.byref(self._chipHeightMM),
            ctypes.byref(self._maxImageSizeX),
            ctypes.byref(self._maxImageSizeY),
            ctypes.byref(self._pixelWidthUM),
            ctypes.byref(self._pixelHeightUM),
            ctypes.byref(self._bpp),
        )

        print(
            f"chipWidthMM: {self._chipWidthMM.value}\n"
            f"chipHeightMM: {self._chipHeightMM.value}\n"
            f"maxImageSizeX: {self._maxImageSizeX.value}\n"
            f"maxImageSizeY: {self._maxImageSizeY.value}\n"
            f"pixelWidthUM: {self._pixelWidthUM.value}\n"
            f"pixelHeightUM: {self._pixelHeightUM.value}\n"
            f"bpp: {self._bpp.value}\n"
        )

        depth = ctypes.c_uint32(bit_depth)

        self._qhyccd.SetQHYCCDBitsMode(self._camera_handle, depth)

        self._qhyccd.SetQHYCCDParam.restype = ctypes.c_uint32
        self._qhyccd.SetQHYCCDParam.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_double
        ]

        if roi is None:
            self._qhyccd.SetQHYCCDResolution(
                self._camera_handle,
                ctypes.c_uint32(0),
                ctypes.c_uint32(0),
                self._maxImageSizeX,
                self._maxImageSizeY
            )
            self._x_size = self._maxImageSizeX.value
            self._y_size = self._maxImageSizeY.value
        else:
            self._qhyccd.SetQHYCCDResolution(
                self._camera_handle,
                ctypes.c_uint32(roi[0][0]),
                ctypes.c_uint32(roi[0][1]),
                ctypes.c_uint32(roi[1][0]),
                ctypes.c_uint32(roi[1][1]),
            )
            self._x_size = roi[1][0]
            self._y_size = roi[1][1]

        self._qhyccd.SetQHYCCDBinMode(
            self._camera_handle,
            ctypes.c_uint32(1),
            ctypes.c_uint32(1)
        )

        self.set_gain_exposure(100, 6000)

        self._qhyccd.ExpQHYCCDSingleFrame(self._camera_handle)

        if bit_depth <= 8:
            self._image_data = np.zeros(
                (self._y_size, self._x_size),
                dtype=np.uint8
            )
        else:
            self._image_data = np.zeros(
                (self._y_size, self._x_size),
                dtype=np.uint16
            )

    def _set_param(self, param: int, value: float):
        r = self._qhyccd.SetQHYCCDParam(
            self._camera_handle,
            ctypes.c_int(param),
            ctypes.c_double(value)
        )
        if r != 0:
            raise RuntimeError(f"param {param}, value {value} returns {r}")

    def set_gain_exposure(self, gain: int, exposure: int):
        self._set_param(6, gain)
        self._set_param(8, exposure)

    def get_frame(self) -> np.ndarray:
        self._qhyccd.ExpQHYCCDSingleFrame(self._camera_handle)
        response = self._qhyccd.GetQHYCCDSingleFrame(
            self._camera_handle,
            ctypes.byref(self._maxImageSizeX),
            ctypes.byref(self._maxImageSizeY),
            ctypes.byref(self._depth),
            ctypes.byref(self._channels),
            self._image_data.ctypes.data_as(ctypes.c_void_p),
        )
        return self._image_data

    def close(self):
        self._qhyccd.CancelQHYCCDExposingAndReadout(self._camera_handle)
        self._qhyccd.CloseQHYCCD(self._camera_handle)
        self._qhyccd.ReleaseQHYCCDResource()


class DummyCamera(Camera):
    def __init__(self, bit_depth: int = 8, roi: Optional[Roi] = None):
        self._bit_depth = bit_depth

        if roi is None:
            self._shape = (64, 64)
        else:
            self._shape = (roi[1][0] - roi[0][0], roi[1][1] - roi[0][1])
        self._gain = 100
        self._exposure = 1000

    def set_gain_exposure(self, gain: int, exposure: int):
        self._gain = gain
        self._exposure = exposure

    def get_frame(self) -> np.ndarray:
        gamma = np.random.gamma(
            shape=self._exposure / 1000,
            scale=2**(self._bit_depth-4),
            size=self._shape
        )
        return np.clip(gamma, 0, 2 ** self._bit_depth)


if __name__ == "__main__":
    # camera = DummyCamera(bit_depth=16)
    camera = QhyCamera(bit_depth=8)
    # camera = QhyCamera(bit_depth=8, roi=((0, 0), (100, 100)))

    camera.set_gain_exposure(gain=100, exposure=666)

    for i in range(1):
        t = time.time()
        frame = camera.get_frame()
        print(f"{time.time()-t:.3f}")

    plt.imshow(frame)
    plt.show()
    plt.hist(frame.flatten(), bins=100)
    plt.show()
