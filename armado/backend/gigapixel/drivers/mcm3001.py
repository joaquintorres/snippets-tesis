from serial import Serial
from serial.serialutil import EIGHTBITS, PARITY_NONE, STOPBITS_ONE
from typing import List

import time


BAUDRATE = 460800
ZFM2020_FACTOR_NM = 211.6667
MAX_STEP_NM = 1e6


class RaceLimitException(Exception):
    pass

class LowerLimitException(RaceLimitException):
    pass

class UpperLimitException(RaceLimitException):
    pass



class MCM3001:
    def __init__(self, port: str):
        """ MCM3001 serial driver

        Args:
            port (str): Connection port of Arduino, like "/dev/ttyUS0"
        """
        if not isinstance(port, str):
            raise TypeError

        self._port = port
        self._baudrate = BAUDRATE
        self._factor_nm = ZFM2020_FACTOR_NM
        self._max_step_nm = MAX_STEP_NM
        self._timeout = 2.0
        self._serial = self._open_serial()

    def _open_serial(self) -> Serial:
        """ Opens a serial port between Arduino and Python """

        return Serial(
            port=self._port, 
            baudrate=self._baudrate,
            bytesize=EIGHTBITS, 
            parity=PARITY_NONE,
            stopbits=STOPBITS_ONE,
            timeout=1
        )
        
    def _send_bytes(self, values: List[int]):
        for x in values:
            self._serial.write(x)

    def _set_encoder_counter(self, stage: int, value: int):
        self._send_bytes([0x09, 0x04, 0x06, 0x00, 0x00, 0x00])
        self._send_bytes([stage, 0x00])
        self._serial.write(value.to_bytes(4, 'little', signed=True))

    def _stop(self, stage: int):
        self._send_bytes([0x65, 0x04, stage, 0x00, 0x00, 0x00])

    def _query_position(self, stage:int) -> float:
        self._send_bytes([0x0A, 0x04, stage, 0x00, 0x00, 0x00])
        
        data = self._serial.read(12)
        
        if data[0:6] != b'\x0B\x04\x06\x00\x00\x00':
            raise RuntimeError(f"Bad response: {data}")
        if stage != int.from_bytes(data[6], 'little'):
            raise RuntimeError(f"Bad response: {data}")
        
        raw_position = int.from_bytes(data[8], 'little')

        return raw_position * self._factor_nm

    def _go_to_position(self, stage: int, position: float):
        raw_position = int(position / self._factor_nm)
        self._send_bytes([0x53, 0x04, 0x06, 0x00, 0x00, 0x00])
        self._send_bytes([stage, 0x00])
        self._serial.write(raw_position.to_bytes(4, 'little', signed=True))

    def _query_is_ready(self, stage: int) -> bool:
        self._send_bytes([0x80, 0x04, stage, 0x00, 0x00, 0x00])
        data = self._serial.read(34)
        return not(data[16] & 0x30)


    def set_zero(self, stage: int):
        self._set_encoder_counter(stage, value=0)

    def move_absolute(self, stage: int, absolute: float):
        pos = self._query_position(stage)
        direction = 1 if pos <= absolute else -1
        while abs(pos - absolute) > self._factor_nm * 2:
            next_step = max(self._max_step_nm, abs(pos - absolute))
            self._go_to_position(stage, pos + direction * next_step)
            t = time.time()
            while not self._query_is_ready(stage):
                if time.time() - t > self._timeout:
                    raise TimeoutError("")
            new_pos = self._query_position(stage)
            
            if abs(new_pos - pos) < self._factor_nm * 2:
                print(f"Limit reached at {new_pos}")
                if direction == 1:
                    raise UpperLimitException()
                else:
                    raise LowerLimitException()
                break

    def move_relative(self, stage:int, relative: float):
        pos = self._query_position(stage)
        self.move_absolute(stage, pos + relative)
