import operator
from collections import defaultdict
from functools import reduce
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def load_data(d_path: Path):
    return np.genfromtxt(d_path, delimiter=";")[:, 0]


def _get_freqs(ys):
    fys = np.sort(np.array(ys).flatten())
    return [((x, y), sum(x >= l and y <= h for l, h in zip(*ys))) for x, y in zip(fys[::2], fys[1::2])]


def find_mode(ys, freqs, *, nmax=3):
    low, high = ys
    if max(low) <= min(high):
        return np.array([max(low), min(high)]),

    zs = defaultdict(list)
    for pt, fr in freqs:
        zs[fr].append(pt)
    return reduce(operator.add, map(zs.get, sorted(zs)[-nmax:]), [])


def find_med(freqs):
    pts, frs = zip(*freqs)
    l, r = 0, sum(frs[1:])
    pivot = 1

    while l < r:
        l += frs[pivot]
        r -= frs[pivot + 1]
        pivot += 1
    return pts[pivot]


def jk(ys):
    low, high = ys
    mil, mal = min(low), max(low)
    mih, mah = min(high), max(high)
    return (mih - mal) / (mah - mil)


def lab3(d_path: Path, img_path: Path, *, tol=1e-4):
    ys = load_data(d_path)
    wys = [ys - tol, ys + tol]
    xs = np.arange(len(ys))
    freqs = _get_freqs(wys)

    modes = find_mode(wys, freqs, nmax=2)
    med = find_med(freqs)
    print(f"jk: {jk(wys)}")
    print(f"med: {med}")
    print(f"modes: {modes}")

    plt.plot(xs, ys, label="orig")
    plt.legend()
    ttl = "original data"
    plt.title(ttl)
    plt.xlabel("n")
    plt.ylabel("mV")
    plt.savefig(img_path.joinpath(ttl))
    plt.show()

    plt.vlines(xs, *wys, label="intervaled")
    plt.legend()
    ttl = "intervaled data"
    plt.title(ttl)
    plt.xlabel("n")
    plt.ylabel("mV")
    plt.savefig(img_path.joinpath(ttl))
    plt.show()

    plt.vlines(xs, *wys, label="intervaled")
    for mode in modes:
        plt.axhline(mode[0], color="gray", linestyle="--", label="mode")
        plt.axhline(mode[1], color="gray", linestyle="--")
    plt.axhline(med[0], color="r", linestyle="--", label="median")
    plt.axhline(med[1], color="r", linestyle="--")
    plt.legend()
    ttl = "intervaled data with mode and median"
    plt.title(ttl)
    plt.xlabel("n")
    plt.ylabel("mV")
    plt.savefig(img_path.joinpath(ttl))
    plt.show()

    pts, frs = map(np.array, zip(*freqs))
    # med to relative frequency
    plt.step(np.sum(pts, axis=1) / 2, frs / len(pts), label="frequency")
    for mode in modes:
        plt.axvline(mode[0], color="gray", linestyle="--", label="mode")
        plt.axvline(mode[1], color="gray", linestyle="--")
    plt.axvline(med[0], color="r", linestyle="--", label="median")
    plt.axvline(med[1], color="r", linestyle="--")
    plt.legend()
    ttl = "frequency histogram"
    plt.title(ttl)
    plt.xlabel("$z_i$")
    plt.xticks(rotation=10)
    plt.ylabel("$\mu_i$")
    plt.savefig(img_path.joinpath(ttl))
    plt.show()


if __name__ == "__main__":
    cwd = Path(__file__).parent
    data_path = cwd.joinpath("data/ch1_800nm_0.04.csv")
    imgs_path = cwd.joinpath("imgs")
    if not imgs_path.exists():
        imgs_path.mkdir(exist_ok=True)

    lab3(data_path, imgs_path)
