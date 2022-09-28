import numpy as np
from intvalpy import Interval
from intvalpy.nonlinear import globopt
from collections import Counter
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import product, filterfalse, count, islice, starmap
from operator import mul, sub, attrgetter, methodcaller, itemgetter
import heapq as hq
from pathlib import Path


def create_imat(mat, epss, eps):
    w, h = map(range, mat.shape)
    return np.array([[Interval(mat[i, j]-eps*epss[i,j], mat[i,j]+eps*epss[i,j]) for j in w] for i in h])


det = lambda m: m[0,0]*m[1,1]-m[0,1]*m[1,0]


int_sign = lambda inter: inter.a > 0 or (-1)*(inter.b < 0)


def special_mat(mat, epss, bounds=np.arange(0, 2, 0.001)):
    if mat.shape != epss.shape:
        raise RuntimeError(f"Wrong matrix shapes: mat: {mat.shape} vs. epps: {epss.shape}")
    return np.array([eps for eps in bounds if not 0 in det(create_imat(mat, epss, eps))])


def baumann(mat, epss, bounds=np.arange(0, 2, .001)):
    if mat.shape != epss.shape:
        raise RuntimeError(f"Wrong matrix shapes: mat: {mat.shape} vs. epps: {epss.shape}")

    def _baumann(eps):
        sign = None
        for n in range(mat.size):
            i, j = divmod(n, len(mat))
            m = create_imat(mat.copy(), epss, eps)
            inter = m[i, j]
            for att in "ab":
                m[i, j] = attrgetter(att)(inter)
                s = int_sign(det(m))
                if sign is not None and sign != s:
                    return False
                sign = s
        return True # means the mat is special

    *sp_eps, = filter(_baumann, bounds)
    # print(sp_eps)
    return np.array(sp_eps)


def rump(mat, epss, bounds=np.arange(0,2,.001)):
    if mat.shape != epss.shape:
        raise RuntimeError(f"Wrong matrix shapes: mat: {mat.shape} vs. epps: {epss.shape}")

    rad = np.vectorize(attrgetter("rad"))
    mid = np.vectorize(attrgetter("mid"))
    typ = methodcaller("astype", np.float64)
    svd = np.linalg.svd

    return np.array([eps for eps in bounds if np.max(svd(typ(rad(A:=create_imat(mat.copy(), epss, eps))))[1]) < np.min(svd(typ(mid(A)))[1])])


def global_opt(im_dir:Path, *, tol=-10, mit=1000):

    def f1(x):
        return .26*(x[0]**2+x[1]**2) - .48*x[0]*x[1]
    f1.name = "Matyas"
    f1.bounds = [-10, 10]
    # f1.extremes = np.array([[0, 0]])
    f1.extremes = np.array([0])


    def f2(x):
        return (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2
    f2.name = "Himmelblau"
    f2.bounds = [-5, 5]
    # f2.extremes = np.array([[3, 2], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]])
    f2.extremes = np.array([0])


    def _glopopt(func, x0, *, tol=10, mit=10000):
        h = []
        b, f = x0.copy, func(x0).a
        tiebreaker = count(step=-1)
        hq.heappush(h, (f, next(tiebreaker), b))

        best = [(func(b).mid, b)]
        it = count()
        while func(b).wid > 10**(tol) and next(it) < mit:
            lside_idx = np.argmax(b.wid)
            lside = b[lside_idx]
            b1, b2 = b.copy, b.copy
            b1[lside_idx] = Interval(lside.a, lside.mid, sortQ=False)
            b2[lside_idx] = Interval(lside.mid, lside.b, sortQ=False)

            hq.heappop(h)
            for _b in (b1, b2):
                hq.heappush(h, (func(_b).a, next(tiebreaker), _b))

            f, _, b = h[0]
            best.append((func(b).mid,b))
        print(next(it) - 1)
        return h, best


    def plt_boxes(boxes):
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle("Boxes and their mids")
        fig.subplots_adjust(left=0.04, bottom=0.04, right=0.99, top=0.955, wspace=0, hspace=0)
        axs[1].set_yticks([])
        def plt_box(box):
            x, y = box
            axs[0].plot([x.a, x.a, x.b, x.b, x.a], [y.a, y.b, y.b, y.a, y.a])
            axs[1].plot(x.mid, y.mid, "o")
        *_, = map(plt_box, boxes)
        # plt.show()
        return fig


    def plt_best_boxes(best, extremes):
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        fig.subplots_adjust(left=0.04, bottom=0.04, right=0.99, top=0.955, wspace=0, hspace=0)
        con, rads = [], []
        for (f, b) in best:
            # if convergence is by the function value
            con.append(np.min(np.linalg.norm(f - extremes[:,np.newaxis], axis=1)))
            # if the convergence is by the mids of the bars
            # con.append(np.min(np.linalg.norm([*starmap(sub, product(b.mid, extremes))], axis=1)))
            rads.append(max(b[0].rad, b[1].rad))
        axs[0].set_title("Rads")
        axs[0].semilogy(rads)
        axs[1].set_title("Convergence")
        axs[1].semilogy(con)
        # plt.show()
        return fig

    N = 2
    for f in (f1, f2):
        a, b = f.bounds
        x = Interval([a]*N, [b]*N)
        h, best = _glopopt(f, x, tol=tol, mit=mit)
        # print(h[0])
        boxes = map(itemgetter(2), h)
        for fig, fname in zip((plt_boxes(boxes),plt_best_boxes(best, f.extremes)), ("boxes", "best_boxes")):
            fig.savefig(im_dir.joinpath(f"{f.name}_{fname}"))


def task():
    mat = np.array([[1, 1], [1.1, 1]])
    epss = np.array([[1, 0], [1, 0]])
    bnds = np.arange(0, 2, 0.001)

    # special_mat(mat, epss, bnds)

    print(baumann(mat, epss, bnds))
    print(rump(mat, epss, bnds))

    im_dir = Path("./imgs")
    if not im_dir.exists():
        im_dir.mkdir()
    global_opt(im_dir, tol=-14, mit=20000)


if __name__ == "__main__":
    task()
