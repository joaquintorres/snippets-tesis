import KYFGLib as ky
import ctypes
import traceback

import numpy as np
import matplotlib.pyplot as plt 
from abc import ABC, abstractmethod
from typing import Tuple, Optional

from time import sleep, time, perf_counter
import sys
import os
import threading
import queue

#Roi = Tuple[int, int]
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

class ImperxCamera(Camera):
    def __init__(self, bit_depth: int = 8, roi: Optional[Roi] = None):
        inicio_init = perf_counter()
        self.grabber_index = 0 #número del grabber de la lista de grabbers
        self.max_boards = 4
        self.max_cams = 4
        self.device_queued_buffers_supported = "FW_DmaCapable_QueuedBuffers_Imp"

        self.init_params = ky.KYFGLib_InitParameters()
        self.connection = -1
        self.stream_info_struct = self.StreamInfoStruct()
        self.cam_handle_array = [[0 for x in range(0)] for y in range(self.max_boards)]
        self.handle = [0 for i in range(self.max_boards)]
        self.camera_stream_handle = 0
        self.frame_data_size = 0
        self.frame_data_aligment = 0
        self.stream_buffer_handle = [0 for i in range(16)]
        self.stream_alligned_buffer = [0 for i in range(16)]
        self._max_image_width = 9344
        self._max_image_height = 7000
        self.n_frames = 1
        self.camera_index = 0
        self.image = 0 
        self._stream_callback_func.__func__.data = 0
        self._stream_callback_func.__func__.copyingDataFlag = 0
        self._time_stamp = 0
        self.image_offset_x = None
        self.image_offset_y = None
        self.image_width = None
        self.image_height = None

        #colas
        self.queue = queue.Queue(10)

        ky.KYFGLib_Initialize(self.init_params)
        inicio_grabber = perf_counter()
        self._connect_to_grabber()
        final_grabber = perf_counter()
        print("grabber: ", final_grabber - inicio_grabber)
        inicio_camera = perf_counter()
        self._connect_to_camera()
        final_camera = perf_counter()
        print("camara: ", final_camera - inicio_camera)
        self._set_roi(roi)

        _, self.exposure =  ky.KYFG_GetCameraValue(self.cam_handle_array[self.grabber_index][0], "ExposureTime")
        _, self.gain = ky.KYFG_GetCameraValue(self.cam_handle_array[self.grabber_index][0], "Gain")

        #La configuración de los bits del adc y del color puede cambiar si se abre la app de vision point
        #si se quiere cambiar el bit depht o el pixel format hay que cambiar el tipo de dato que aparece
        #en la función _stream_callback_func en data = np.frombuffer(buffer_byte_array, dtype=np.int16)
        # a data = np.frombuffer(buffer_byte_array, dtype=np.uint8)
        ky.KYFG_SetCameraValue(self.cam_handle_array[self.grabber_index][0], "AdcBitDepth", "Bit12")
        ky.KYFG_SetCameraValue(self.cam_handle_array[self.grabber_index][0], "PixelFormat", "Mono12")
        self.bit_depth = ky.KYFG_GetCameraValue(self.cam_handle_array[self.grabber_index][0], "AdcBitDepth")
        self.pixel_format = ky.KYFG_GetCameraValue(self.cam_handle_array[self.grabber_index][0], "PixelFormat")
        print("Bit Depth ADC:",self.bit_depth)
        print("Bit PixelFormat:",self.pixel_format)

        _, self.camera_stream_handle = ky.KYFG_StreamCreate(self.cam_handle_array[self.grabber_index][0], 0)

        _, = ky.KYFG_StreamBufferCallbackRegister(self.camera_stream_handle, self._stream_callback_func, ky.py_object(self.stream_info_struct))

        self._allocate_memory()

        final_init = perf_counter()
        print("tiempo init: ", final_init - inicio_init)

    def _connect_to_grabber(self):
        _, fg_amount = ky.KY_DeviceScan()
        _, dev_info = ky.KY_DeviceInfo(self.grabber_index)
        try:
            self.connection = self._set_grabber_connection()
        except ky.KYException as err:
            print("Error al conectar con el frame grabber")
        return



    def _set_grabber_connection(self):
        self.handle
        connected_fghandle, = ky.KYFG_Open(self.grabber_index)
        connected = connected_fghandle.get()
        self.handle[self.grabber_index] = connected
        ky.KYFG_GetGrabberValueInt(self.handle[self.grabber_index], self.device_queued_buffers_supported)
        return 0

    def _connect_to_camera(self):
        _, self.cam_handle_array[self.grabber_index] = ky.KYFG_UpdateCameraList(self.handle[self.grabber_index])
        cams_num = len(self.cam_handle_array[self.grabber_index])
        if (cams_num < 1):
            print("No se encontraron camaras")
        KYFG_CameraOpen2_status, = ky.KYFG_CameraOpen2(self.cam_handle_array[self.grabber_index][0], None)
        if KYFG_CameraOpen2_status == ky.FGSTATUS_OK:
            print("Camara conectada correctamente")

    def _set_roi(self, roi):
        if roi is None:
            _, = ky.KYFG_SetCameraValueInt(self.cam_handle_array[self.grabber_index][0], "Width", self._max_image_width)
            _, = ky.KYFG_SetCameraValueInt(self.cam_handle_array[self.grabber_index][0], "Height", self._max_image_height)
            self.image_offset_x = 0
            self.image_offset_y = 0
            self.image_width = self._max_image_width
            self.image_height = self._max_image_height


        else: 
            self.image_offset_x = roi[0][0]
            self.image_offset_y = roi[0][1]
            self.image_width = roi[1][0] - roi[0][0]
            self.image_height = roi[1][1] - roi[0][1]
            print("height", type(self.image_height), self.image_height)
            print("width", type(self.image_width), self.image_width)
            print("offset x", type(self.image_offset_x), self.image_offset_x)
            print("offset y ", type(self.image_offset_y), self.image_offset_y)
            _, = ky.KYFG_SetCameraValueInt(self.cam_handle_array[self.grabber_index][0], 'OffsetX', self.image_offset_x)
            _, = ky.KYFG_SetCameraValueInt(self.cam_handle_array[self.grabber_index][0], 'OffsetY', self.image_offset_y)
            _, = ky.KYFG_SetCameraValueInt(self.cam_handle_array[self.grabber_index][0], "Width", self.image_width)
            _, = ky.KYFG_SetCameraValueInt(self.cam_handle_array[self.grabber_index][0], "Height", self.image_height)
            

    def _allocate_memory(self):
        _, payload_size, _, _ = ky.KYFG_StreamGetInfo(self.camera_stream_handle, ky.KY_STREAM_INFO_CMD.KY_STREAM_INFO_PAYLOAD_SIZE)

        _, buf_allignment, _, _ = ky.KYFG_StreamGetInfo(self.camera_stream_handle, ky.KY_STREAM_INFO_CMD.KY_STREAM_INFO_BUF_ALIGNMENT)

        for iFrame in range(len(self.stream_buffer_handle)):
            self.stream_alligned_buffer[iFrame] = ky.aligned_array(buf_allignment, ky.c_ubyte, payload_size)
            _, self.stream_buffer_handle[iFrame] = ky.KYFG_BufferAnnounce(self.camera_stream_handle, self.stream_alligned_buffer[iFrame], None)


    def _stream_callback_func(self, buff_handle, user_context):


        if buff_handle == 0:
            self._stream_callback_func.__func__.copyingDataFlag = 0
            return


        _, self._time_stamp, _, _ =  ky.KYFG_BufferGetInfo(buff_handle, ky.KY_STREAM_BUFFER_INFO_CMD.KY_STREAM_BUFFER_INFO_TIMESTAMP)

        stream_info = ky.cast(user_context, ky.py_object).value
        stream_info.callbackCount = stream_info.callbackCount + 1

        _, pointer, _, _ = ky.KYFG_BufferGetInfo(buff_handle, ky.KY_STREAM_BUFFER_INFO_CMD.KY_STREAM_BUFFER_INFO_BASE)
        _, buffer_size, _, _ = ky.KYFG_BufferGetInfo(buff_handle, ky.KY_STREAM_BUFFER_INFO_CMD.KY_STREAM_BUFFER_INFO_SIZE)

        buffer_byte_array = bytearray(ctypes.string_at(pointer, buffer_size))
        #data = np.frombuffer(buffer_byte_array, dtype=np.ubyte)
        data = np.frombuffer(buffer_byte_array, dtype=np.uint16)
        self.image = data.reshape(self.image_height, self.image_width)
        self.queue.put(self.image)


        if self._stream_callback_func.__func__.copyingDataFlag == 0:
           self._stream_callback_func.__func__.copyingDataFlag = 1

        sys.stdout.flush()
        self._stream_callback_func.__func__.copyingDataFlag = 0
        return 

    def _start_camera(self):
        # put all buffers to input queue
        KYFG_BufferQueueAll_status, = ky.KYFG_BufferQueueAll(self.camera_stream_handle, ky.KY_ACQ_QUEUE_TYPE.KY_ACQ_QUEUE_UNQUEUED, ky.KY_ACQ_QUEUE_TYPE.KY_ACQ_QUEUE_INPUT)
        
        KYFG_CameraStart_status, = ky.KYFG_CameraStart(self.cam_handle_array[self.grabber_index][self.camera_index], self.camera_stream_handle, self.n_frames)
        return 0

    def set_gain_exposure(self, gain, exposure):
        _,  = ky.KYFG_SetCameraValue(self.cam_handle_array[self.grabber_index][0], "ExposureTime", exposure)
        _, self.exposure =  ky.KYFG_GetCameraValue(self.cam_handle_array[self.grabber_index][0], "ExposureTime")
        _,  = ky.KYFG_SetCameraValue(self.cam_handle_array[self.grabber_index][0], "Gain", gain)
        _, self.gain =  ky.KYFG_GetCameraValue(self.cam_handle_array[self.grabber_index][0], "Gain")
        
        return

    def get_frame(self):
        sys.stdout.flush()


        self._start_camera()

        self.image = self.queue.get()

        #while self._stream_callback_func.__func__.copyingDataFlag == 0:
        #    pass
        self._stream_callback_func.__func__.copyingDataFlag = 1
        sys.stdout.flush()
        _, = ky.KYFG_CameraStop(self.cam_handle_array[self.grabber_index][0])

        return self.image

    def close(self):

        _, = ky.KYFG_CameraClose(self.cam_handle_array[self.grabber_index][0])
        _, = ky.KYFG_Close(self.handle[self.grabber_index])

    class StreamInfoStruct:
        def __init__(self):
            self.width = 640
            self.height = 480
            self.callbackCount = 0
            return
