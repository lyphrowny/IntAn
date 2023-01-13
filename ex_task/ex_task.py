from functools import partial

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import sympy as sp
from sympy import print_latex
from sympy.abc import x as spx

from lab4.lab4 import load_data, load_weights, find_pts, plot_poly, predict_plot, lines_plot, weighted_plot
from lab3.lab3 import _get_freqs, find_med, find_mode, jk


def find_all_plot(xs, wys, *, i_path: Path, detrend=False):
    _detr_str = "detrend "
    freqs = _get_freqs(wys)

    modes = find_mode(wys, freqs, nmax=2)
    med = find_med(freqs)
    print(f"jk: {jk(wys)}")
    print(f"med: {med}")
    print(f"modes: {modes}")

    plt.vlines(xs, *wys, label="intervaled")
    plt.legend()
    ttl = detrend * _detr_str + "intervaled data"
    plt.title(ttl)
    plt.xlabel("n")
    plt.ylabel("mV")
    plt.savefig(i_path.joinpath(ttl))
    plt.show()

    plt.vlines(xs, *wys, label="intervaled")
    for mode in modes:
        plt.axhline(mode[0], color="gray", linestyle="--", label="mode")
        plt.axhline(mode[1], color="gray", linestyle="--")
    plt.axhline(med[0], color="r", linestyle="--", label="median")
    plt.axhline(med[1], color="r", linestyle="--")
    plt.legend()
    ttl = detrend * _detr_str + "intervaled data with mode and median"
    plt.title(ttl)
    plt.xlabel("n")
    plt.ylabel("mV")
    plt.savefig(i_path.joinpath(ttl))
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
    ttl = detrend * _detr_str + "frequency histogram"
    plt.title(ttl)
    plt.xlabel("$z_i$")
    plt.xticks(rotation=10)
    plt.ylabel("$\mu_i$")
    plt.savefig(i_path.joinpath(ttl))
    plt.show()


def lab3_ext(d_path: Path, w_path: Path, *, i_path: Path, tol=1.9 * 1e-4):
    ys = load_data(d_path)
    xs = np.arange(len(ys))
    max_weight = np.max(load_weights(w_path))
    low, high = ys - tol * max_weight, ys + tol * max_weight

    # pts = find_pts(xs, (low, high))
    pts = [[7.98265896e-06, 2.35179375e-02], [1.63799716e-05, 2.20143629e-02], [7.54437870e-06, 2.22187311e-02],
           [1.35555556e-05, 2.25538264e-02]]
    *polys, = map(np.poly1d, pts)
    pre_trend = np.zeros((len(polys), len(xs)))
    for i, poly in enumerate(polys):
        pre_trend[i] = poly(xs)
    low_trend = np.min(pre_trend, axis=0)
    high_trend = np.max(pre_trend, axis=0)

    low_detrend = low - high_trend
    high_detrend = high - low_trend

    plt.plot(xs, ys, label="orig")
    plt.legend()
    ttl = "original data"
    plt.title(ttl)
    plt.xlabel("n")
    plt.ylabel("mV")
    plt.savefig(i_path.joinpath(ttl))
    plt.show()

    find_all_plot(xs, (ys - tol, ys + tol), i_path=i_path)
    find_all_plot(xs, (low_detrend, high_detrend), i_path=i_path, detrend=True)


def lab4_ext(predict_at, d_path: Path, w_path: Path, *, i_path: Path, tol=1.9 * 1e-4, trslice=slice(50, 150)):
    ys = load_data(d_path)
    weights = load_weights(w_path)
    xs = np.arange(len(ys))

    # weighted ys
    wys = [ys - tol * weights, ys + tol * weights]
    # max weighted ys
    max_weight = max(weights)
    mys = [ys - tol * max_weight, ys + tol * max_weight]
    train = [(ys - tol * max_weight)[trslice], (ys + tol * max_weight)[trslice]]

    pts = find_pts(xs[trslice], train)
    # pts = np.array([(7.982658959537573e-06, 0.023517937499999995), (1.6379971590909073e-05, 0.022014362926136365),
    #        (7.544378698224811e-06, 0.022218731139053267), (1.3555555555555745e-05, 0.022553826388888847)])
    *polys, = map(np.poly1d, pts)

    lines_plot(xs, mys, polys, save_path=i_path)

    # pretty printing polys
    print("polys")
    sp.init_printing()
    for poly in polys:
        print_latex(sp.Poly(reversed(poly.coef), spx).as_expr())
    print("\ncenter line coeffs")
    print_latex(sp.Poly(reversed(np.poly1d(pts.mean(axis=0)).coef), spx).as_expr())
    print()

    print("predicted vals")
    for pt in predict_at:
        predict_plot(pt, [(ys - tol * max_weight)[pt], (ys + tol * max_weight)[pt]], pt, polys, save_path=i_path)

    plot_poly(pts, save_path=i_path)


if __name__ == "__main__":
    cwd = Path(__file__).parent
    data_path = Path("./data/ch1_800nm_0.04.csv")
    weights_path = Path("./data/ch1_800nm_0.04.txt")
    imgs_path = Path("./imgs")

    if not imgs_path.exists():
        imgs_path.mkdir(parents=True, exist_ok=True)

    # predict_at = (-10, 100.5, 1000)
    predict_at = (0, 101, 199)

    # lab3_ext(cwd.joinpath(data_path), cwd.joinpath(weights_path), i_path=cwd.joinpath(imgs_path))
    lab4_ext(predict_at, cwd.joinpath(data_path), cwd.joinpath(weights_path), i_path=cwd.joinpath(imgs_path))
