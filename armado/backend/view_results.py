import numpy as np
import pathlib
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from chanoscope import FpmAcquisitionSetup

path = pathlib.Path("~/chanodata/").expanduser()

histograms = {
    (x, y): np.load(str(path / f"exp_{x}_{y}_rgb.npz"))["hist"]
    for x, y in zip(range(32), range(32))
}

overexposed = np.asarray([
    [
        np.load(str(path / f"exp_{x}_{y}_rgb.npz"))["p50"]
        for x in range(32)
    ]
    for y in range(32)
])

plt.plot(histograms[(15, 15)], marker='o', ls='')
plt.show()

plt.imshow(overexposed, origin="lower")
plt.show()

xx, yy = np.meshgrid(np.arange(32), np.arange(32))


def gauss(xx, yy, a, sigma, x0, y0):
    return a * np.exp(-((xx-x0)**2+(yy-y0)**2)/(2*sigma**2))


def error(params):
    return (gauss(xx, yy, *params) - overexposed).flatten()


result = least_squares(error, x0=np.asarray([8e4, 7.0, 15, 15]))

print(result.x)

fit = gauss(xx, yy, *result.x)
plt.imshow(fit.max() / fit, origin="lower")
plt.show()

np.save("exposure_matrix.npy", fit.max() / fit)
