from typing import Tuple, Callable

from matplotlib.figure import Figure, Axes
import numpy as np

from .config import DATA_BIT_DEPTH

M = 32
Scheme = Callable[[int, int, int], bool]


def square(x: int, y: int, size: int) -> bool:
    return bool(max(abs(x), abs(y)) < size // 2)


def circle(x: int, y: int, size: int) -> bool:
    return bool(np.sqrt(x ** 2 + y ** 2) < size // 2)


def diamond(x: int, y: int, size: int) -> bool:
    return bool(abs(x) + abs(y) < size // 2)


def get_scheme(name: str) -> Scheme:
    return {
        "square": square,
        "circle": circle,
        "diamond": diamond,
    }[name]


def xy_iterator(size: int, center: Tuple[int, int] = (15, 15),
                scheme: Scheme = square) -> Tuple[int, int]:
    yield 0 + center[0], 0 + center[1]
    for i in range(M + 1):
        n = np.arange(i * 4)
        a = i - np.abs(n - i * 2)
        b = np.roll(a, i)
        for x, y in zip(a, b):
            if -M // 2 < x <= M // 2 and -M // 2 < y <= M // 2:
                if scheme(x, y, size) is True:
                    yield x + center[0], y + center[1]


def fig_histogram(frame: np.ndarray, color: str, title: str="") -> Figure:
    histogram, bins = np.histogram(frame.flatten(), bins=100, range=(0, 2**DATA_BIT_DEPTH-1))
    x = (np.arange(100) / 100)[:-1]
    y = histogram[:-1]

    fig: Figure = Figure(figsize=(4, 2))
    ax: Axes = fig.subplots()
    ax.fill_between(
        x, 0, y, where=x <= 0.05,
        color='red', alpha=0.25
    )
    ax.fill_between(
        x, 0, y, where=(x >= 0.05) & (x <= 0.25),
        color='yellow', alpha=0.25
    )
    ax.fill_between(
        x, 0, y, where=(x >= 0.25) & (x <= 0.75),
        color='green', alpha=0.25
    )
    ax.fill_between(
        x, 0, y, where=(x >= 0.75) & (x <= 0.95),
        color='yellow', alpha=0.25
    )
    ax.fill_between(
        x, 0, y, where=(x >= 0.95),
        color='red', alpha=0.25
    )

    ax.plot(x, y, ls='', marker='o', markersize=2, color=color)
    ax.set_xticks([0.05, 0.25, 0.75, 0.95])
    ax.set_xticklabels(["5%", "25%", "75%", "95%"])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    total = np.sum(histogram)
    red_count = (np.sum(histogram[0:5]) + np.sum(histogram[95:])) / total
    yellow_count = (np.sum(histogram[5:25]) + np.sum(histogram[75:95])) / total
    green_count = np.sum(histogram[25:75]) / total

    ax.bar(1.1, red_count, width=.25,
           transform=ax.transAxes, color='red', alpha=0.5)
    ax.bar(1.1, yellow_count, width=.25, bottom=red_count,
           transform=ax.transAxes, color='yellow', alpha=0.5)
    ax.bar(1.1, green_count, width=.25, bottom=red_count + yellow_count,
           transform=ax.transAxes, color='green', alpha=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(0, 1.1)
    fig.tight_layout()
    return fig


def fig_camera(frame: np.ndarray, color: str) -> Figure:
    fig: Figure = Figure(figsize=(frame.shape[1] / 1000, frame.shape[0] / 1000))
    ax: Axes = fig.subplots()

    ax.imshow(frame, cmap="gray", vmin=0, vmax=2**DATA_BIT_DEPTH-1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    fig.tight_layout()

    return fig


def fig_scheme(size, scheme):
    fig = Figure(figsize=(6, 6))
    ax = fig.subplots()
    points = np.array([[x, y] for x, y in xy_iterator(32)])
    points2 = np.array([[x, y] for x, y in xy_iterator(size, scheme=scheme)])
    ax.plot(points[:, 0], points[:, 1], marker='o', ls='', lw=2, alpha=0.6,
            mec="k", markersize=8, color='w')
    ax.plot(points2[:, 0], points2[:, 1], marker='o', ls='', lw=2, alpha=0.6,
            mec="k", markersize=8, color='y')
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def f_scheme(size: int, scheme: Scheme) -> np.ndarray:
    return np.array([[x, y] for x, y in xy_iterator(size, scheme=scheme)])
