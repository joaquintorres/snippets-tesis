import numpy as np
import numba as nb
from numba.experimental import jitclass
   
spec = [
    ('_cursor', nb.intp),
    ('_count', nb.intp),
    ('_buffer', nb.intp[::1]),
]

@jitclass(spec)
class Deque1D(object):
    def __init__(self, buffer_size=1024):
        self._cursor = 0
        self._count = 0
        self._buffer = np.zeros(buffer_size, dtype=np.intp)

    @property
    def cursor(self):
        return self._cursor

    @property
    def count(self):
        return self._count

    def popleft(self):
        if self._count == 0:
            raise ValueError("No elements in buffer")

        value = self._buffer[self._cursor]
        if self._cursor == self._buffer.size - 1:
            self._cursor = 0
        else:
            self._cursor += 1

        self._count -= 1

        return value

    def append(self, value):
        if self._count == self._buffer.size:
            raise ValueError("Buffer full.")
        self._buffer[(self._cursor + self._count) % self._buffer.size] = value
        self._count += 1

spec2d = [
    ('_cursor', nb.intp),
    ('_count', nb.intp),
    ('_buffer', nb.intp[:, :]),
]

@jitclass(spec2d)
class Deque2D(object):
    def __init__(self, buffer_size=1024):
        self._cursor = 0
        self._count = 0
        self._buffer = np.zeros((buffer_size, 2), dtype=np.intp)

    @property
    def cursor(self):
        return self._cursor

    @property
    def count(self):
        return self._count

    def popleft(self):
        if self._count == 0:
            raise ValueError("No elements in buffer")

        value = self._buffer[self._cursor]
        if self._cursor == self._buffer.shape[0] - 1:
            self._cursor = 0
        else:
            self._cursor += 1

        self._count -= 1

        return value

    def append(self, value):
        if self._count == self._buffer.shape[0]:
            raise ValueError("Buffer full.")
        self._buffer[(self._cursor + self._count) % self._buffer.shape[0]] = value
        self._count += 1
