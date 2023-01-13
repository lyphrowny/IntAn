from itertools import islice
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy import print_latex
from sympy.abc import x as spx


def load_data(d_path: Path):
    return np.genfromtxt(d_path, delimiter=";")[:, 0]


def load_weights(w_path: Path):
    return np.genfromtxt(w_path)[1:]


def find_pts(xs, mys):
    low, high = mys
    pts = []
    for i, (x_i, l_i, h_i) in enumerate(zip(xs, *mys)):
        for x_j, l_j, h_j in islice(zip(xs, low, high), i + 1, None):
            for pair in (
                    (l_i, l_j),
                    (l_i, h_j),
                    (h_i, l_j),
                    (h_i, h_j)
            ):
                k, b = np.polyfit((x_i, x_j), pair, 1)
                _ys = k * xs + b
                if np.all(np.logical_and(low <= _ys, _ys <= high)):
                    pts.append((k, b))
    return np.array(pts)


def weighted_plot(xs, ys, ttl, *, save_path: Path):
    plt.vlines(xs, *ys, label=ttl)
    plt.legend()
    plt.title(ttl)
    plt.xlabel("n")
    plt.ylabel("mV")
    plt.savefig(save_path.joinpath(ttl))
    plt.show()


def lines_plot(xs, mys, polys, *, save_path: Path):
    plt.vlines(xs, *mys, label="weighted data", colors="gray")
    for i, poly in enumerate(polys):
        plt.plot(xs, poly(xs), label=f"line #{i}")
    plt.legend()
    ttl = "weighted data with found lines"
    plt.title(ttl)
    plt.xlabel("n")
    plt.ylabel("mV")
    plt.savefig(save_path.joinpath(ttl))
    plt.show()


def predict_plot(xs, ys, pt, polys, *, save_path: Path, l=10, r=11, step=1):
    mi, *_, ma = sorted([poly(pt) for poly in polys])
    _xs = np.arange(pt - l, pt + r, step)

    print(f"""&y({pt}) = [{mi},\; {ma}] \\\\
&mid = {(mi + ma) / 2}, \; rad = {ma - mi}""")

    for i, poly in enumerate(polys):
        plt.plot(_xs, poly(_xs), label=i)
    plt.vlines(xs, *ys, colors="gray", linewidth=10, label="orig data")
    # print(pt, ys)
    plt.vlines(pt, mi, ma, linestyle="dashed")
    plt.scatter(pt, (mi + ma) / 2, facecolors="none", edgecolors="r", s=75, label=f"predicted at {pt}", lw=3, zorder=100)
    ttl = f"prediction at {pt}"
    plt.title(ttl)
    plt.xlabel("n")
    plt.ylabel("mV")
    plt.legend()
    plt.savefig(save_path.joinpath(ttl + ".png"))
    plt.show()


def plot_poly(pts, *, save_path: Path):
    y, x = map(np.array, zip(*pts))
    plt.scatter(x, y, edgecolors="g", s=75, lw=3)
    order = np.argsort(np.arctan2(y - y.mean(), x - x.mean()))
    plt.fill(x[order], y[order], "g", alpha=.5)
    ttl = "information set"
    plt.title(ttl)
    plt.xlabel("k")
    plt.xticks(rotation=10)
    plt.ylabel("b")
    plt.savefig(save_path.joinpath(ttl))
    plt.show()


def lab4(predict_at, d_path: Path, w_path: Path, img_path: Path, tol=1e-4 * 1.9):
    ys = load_data(d_path)
    weights = load_weights(w_path)
    xs = np.arange(len(ys))

    # weighted ys
    wys = [ys - tol * weights, ys + tol * weights]
    # max weighted ys
    mys = [ys - tol * max(weights), ys + tol * max(weights)]

    for _ys, ttl in zip((wys, mys), ("weighted data", "max weighted data")):
        weighted_plot(xs, _ys, ttl, save_path=img_path)

    pts = find_pts(xs, mys)
    # pts = np.array([(7.982658959537573e-06, 0.023517937499999995), (1.6379971590909073e-05, 0.022014362926136365),
    #        (7.544378698224811e-06, 0.022218731139053267), (1.3555555555555745e-05, 0.022553826388888847)])
    *polys, = map(np.poly1d, pts)

    lines_plot(xs, mys, polys, save_path=img_path)

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
        predict_plot(pt, [(ys - tol * max(weights))[pt], (ys + tol * max(weights))[pt]], pt, polys, save_path=img_path)

    plot_poly(pts, save_path=img_path)


if __name__ == "__main__":
    cwd = Path(__file__).parent

    data_path = Path("./data/ch1_800nm_0.04.csv")
    weights_path = Path("./data/ch1_800nm_0.04.txt")
    imgs_path = Path("./imgs")
    if not imgs_path.exists():
        imgs_path.mkdir(parents=True, exist_ok=True)

    predict_at = (-10, 100.5, 1000)
    predict_at = (0, 101, 199)

    lab4(predict_at, cwd.joinpath(data_path), cwd.joinpath(weights_path), imgs_path)
