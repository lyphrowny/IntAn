import intvalpy as ip
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

save_path = Path(__file__).parent.joinpath("imgs")
if not save_path.exists():
    save_path.mkdir(exist_ok=True)

def line(ax, coefs, res, x, y, mod):
    if coefs[0] == 0 and coefs[1] == 0:
        return
    if coefs[1] == 0:
        ax.plot([res / coefs[0]] * len(y), y, mod)
    else:
        ax.plot(x, (res - coefs[0] * x) / coefs[1], mod)


def start_linear_system_plot(A, b):
    colors = ['r', 'g', 'b', 'k']
    x = np.linspace(-1, 5, 100)
    y = [-0.5, 4]
    fig, ax = plt.subplots()
    for coefs, res, color in zip(A, b, colors):
        line(ax, coefs.mid, res.mid, x, y, color + '-')


def linear_system_plot(A, b, title):
    start_linear_system_plot(A, b)
    plt.title(title)
    plt.grid()
    global save_path
    plt.savefig(save_path.joinpath(title.replace("\n", " ").replace(" ", "_")))
    plt.show()


def start_tol_plot(A, b, needVe=False):
    x, y = np.mgrid[-1:5:100j, -0.5:3:45j]
    z = np.zeros(x.shape)
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            z[i][j] = ip.linear.Tol(A, b, [x[i][j], y[i][j]])
    max = ip.linear.Tol(A, b, maxQ=True)

    fig, ax = plt.subplots()
    cs = ax.contour(x, y, z, levels = 20)
    fig.colorbar(cs, ax=ax)
    ax.clabel(cs)
    ax.plot(max[1][0], max[1][1], 'r*', label='Максимум ({:.6f}, {:.6f}),\nзначение: {:.6f}'.format(max[1][0], max[1][1], max[2]))
    if needVe:
        ive = ip.linear.ive(A, b)
        rve = ive * np.linalg.norm(b.mid) / np.linalg.norm([max[1][0], max[1][1]])
        print("ive: {}".format(ive))
        print("rve: {}".format(rve))
        iveRect = plt.Rectangle((max[1][0] - ive, max[1][1] - ive), 2 * ive, 2 * ive, edgecolor='red', facecolor='none', label='Брус ive')
        plt.gca().add_patch(iveRect)
        rveRect = plt.Rectangle((max[1][0] - rve, max[1][1] - rve), 2 * rve, 2 * rve, edgecolor='blue', facecolor='none', label='Брус rve')
        plt.gca().add_patch(rveRect)
    return max[2]


def start_tol_plot_with_lines(A, b, needVe=False):
    x, y = np.mgrid[-1:5:200j, -0.5:4:90j]
    z = np.zeros(x.shape)
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            z[i][j] = ip.linear.Tol(A, b, [x[i][j], y[i][j]])
    max = ip.linear.Tol(A, b, maxQ=True)

    fig, ax = plt.subplots()
    cs = ax.contour(x, y, z, levels = 20)
    fig.colorbar(cs, ax=ax)
    ax.clabel(cs)
    ax.plot(max[1][0], max[1][1], 'r*', label='Максимум ({:.6f}, {:.6f}),\nзначение: {:.6f}'.format(max[1][0], max[1][1], max[2]))
    if needVe:
        ive = ip.linear.ive(A, b)
        rve = ive * np.linalg.norm(b.mid) / np.linalg.norm([max[1][0], max[1][1]])
        print("ive: {}".format(ive))
        print("rve: {}".format(rve))
        iveRect = plt.Rectangle((max[1][0] - ive, max[1][1] - ive), 2 * ive, 2 * ive, edgecolor='red', facecolor='none', label='Брус ive')
        plt.gca().add_patch(iveRect)
        rveRect = plt.Rectangle((max[1][0] - rve, max[1][1] - rve), 2 * rve, 2 * rve, edgecolor='blue', facecolor='none', label='Брус rve')
        plt.gca().add_patch(rveRect)

    colors = ['r', 'g', 'b', 'k']
    x = np.linspace(-1, 5, 100)
    y = [-0.5, 4]
    for coefs, res, color in zip(A, b, colors):
        line(ax, coefs.mid, res.mid, x, y, color + '-')

    return max[2]


def end_plot(title):
    global save_path
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(save_path.joinpath(title.replace("\n", " ").replace(" ", "_")))
    plt.show()


def tol_plot(A, b, title, needVe=False):
    tol = start_tol_plot(A, b, needVe)
    end_plot(title)
    return tol


def b_correction(b, K, weights):
    return b + K * ip.Interval(-1, 1) * weights


def A_correction(A, K, weights, E):
    mul = K * weights * E
    newA = A.a - mul.a
    newB = A.b - mul.b
    return ip.Interval(newA, newB)


# midA = np.array([[1, 1.5], [1, -2], [1, 0], [0, 1]])
# radA = np.array([[0.5, 1], [0, 1], [0.1, 0], [0, 0.1]])
midA = np.array([[1, 2], [1, -3], [1, 0], [0, 1]])
radA = np.array([[1, 1], [0, 1], [0.25, 0], [0, 0.25]])
A = ip.Interval(midA, radA, midRadQ=True)

# b1, b2 = np.random.uniform(1, 5), np.random.uniform(1, 5)
b1, b2 = 4, 1
# b1, b2 = 3.2, 1.8
print("b values: {}, {}".format(b1, b2))
midb = np.array([5, 0, b1, b2])
radb = np.array([2, 0.5, 1, 1])
b = ip.Interval(midb, radb, midRadQ=True)

linear_system_plot(A, b, 'mid of SLE')
maxTol = tol_plot(A, b, 'Tol for SLE')
K = 1.5 * maxTol
print("K: {}".format(K))

weightsB = np.ones(len(b))
bCorrected = b_correction(b, K, weightsB)
print("corrected b: {}".format(bCorrected))
maxTolBCorrected = tol_plot(A, bCorrected, 'Tol for SLE\ncorrected right part', True)

midE = np.zeros((4, 2))
radE = np.array([[0.3, 0.6], [0, 0.6], [0.06, 0], [0, 0.06]])
E = ip.Interval(midE, radE, midRadQ=True)
weightsA = np.ones((4, 2))
ACorrected = A_correction(A, K, weightsA, E)
print("corrected A: {}".format(ACorrected))
maxTolACorrected = tol_plot(ACorrected, b, 'Tol for SLE\ncorrected left part', True)

start_tol_plot_with_lines(ACorrected, b)
end_plot('Tol and mid for SLE\ncorrected left part')

ACorrected1 = ip.Interval(midA, radA, midRadQ=True)
row = 0
ACorrected1[row, 0] = ip.Interval(midA[row, 0], midA[row, 0])
ACorrected1[row, 1] = ip.Interval(midA[row, 1], midA[row, 1])
start_tol_plot_with_lines(ACorrected1, b)
end_plot('Tol and mid for SLE\ncorrected row (1)')
print("corrected A1: {}".format(ACorrected1))

ACorrected2 = ip.Interval(midA, radA, midRadQ=True)
row = 1
ACorrected2[row, 0] = ip.Interval(midA[row, 0], midA[row, 0])
ACorrected2[row, 1] = ip.Interval(midA[row, 1], midA[row, 1])
start_tol_plot_with_lines(ACorrected2, b)
end_plot('Tol and mid for SLE\ncorrected row (2)')
print("corrected A2: {}".format(ACorrected2))

ACorrected3 = ip.Interval(midA, radA, midRadQ=True)
row = 2
ACorrected3[row, 0] = ip.Interval(midA[row, 0], midA[row, 0])
ACorrected3[row, 1] = ip.Interval(midA[row, 1], midA[row, 1])
start_tol_plot_with_lines(ACorrected3, b)
end_plot('Tol and mid for SLE\ncorrected row (3)')
print("corrected A3: {}".format(ACorrected3))

ACorrected4 = ip.Interval(midA, radA, midRadQ=True)
row = 3
ACorrected4[row, 0] = ip.Interval(midA[row, 0], midA[row, 0])
ACorrected4[row, 1] = ip.Interval(midA[row, 1], midA[row, 1])
start_tol_plot_with_lines(ACorrected4, b)
end_plot('Tol and mid for SLE\ncorrected row (4)')
print("corrected A4: {}".format(ACorrected4))