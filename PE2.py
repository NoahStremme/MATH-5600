
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cholesky(A):

    n = len(A)
    l = [[0.0]*n for i in range(n)]
    for k in range(n):
        sum = A[k][k]
        for j in range(k):
            sum -= l[k][j] * l[k][j]

        if sum <= 0:
            raise ValueError("Matrix is not numerically positive definite") 

        l[k][k] = sqrt(sum)

        for i in range(k+1, n):
            sum = A[i][k]
            for j in range(k):
                sum -= l[k][j] * l[i][j]
            l[i][k] = sum / l[k][k]

    return l


def solve_cholesky(l, y):
    n = len(l)

    z = [0.0]*n
    for i in range(n):
        z[i] = (y[i] - sum(l[i][j]*z[j] for j in range(i)))/ l[i][i]

    x = [0.0]*n
    for i in range(n-1, -1, -1):
        x[i] = (z[i] - sum(l[j][i]*x[j] for j in range(i+1, n))) / l[i][i]
    
    return x


def gram_schmidt(A):
    A = np.array(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n), dtype=float)
    R = np.zeros((n, n), dtype=float)

    for k in range(n):
        v = A[:, k].copy()

        for j in range(k):
            R[j, k] = Q[:,j] @ A[:,k]
            v = v - R[j, k] * Q[:, j]

        R[k, k] = np.linalg.norm(v)
        Q[:, k] = v / R[k,k]

    return Q, R


def solve_qr(Q, R, y):
    b= Q.T @ y
    n = R.shape[1]

    x = np.zeros(n)

    for i in range(n-1, -1, -1):
        x[i] = (b[i] - sum(R[i,j]*x[j] for j in range(i+1,n))) / R[i,i]

    return x


def ls_normal_eq(A, y):
    A = np.array(A, dtype=float)
    y = np.array(y, dtype=float)

    AtA = (A.T @ A).tolist()
    Aty = (A.T @ y).tolist()

    l = cholesky(AtA)
    x = solve_cholesky(l, Aty)
    return x


def ls_qr(A, y):
    Q, R = gram_schmidt(A)
    x = solve_qr(Q, R, y)
    return x


def eval_poly(c, t):
    y = np.zeros(len(t))
    for k in range(len(c)):
        y += c[k] * t**k
    return y


def hilbert(n):
    return [[1/(i+j+1) for j in range(n)] for i in range(n)]


def vandermonde(t, deg):
    return [[ti**k for k in range(deg+1)] for ti in t]


def local_ls_derivative(x, y, W, deg):
    centers = []
    derivs = []

    for i in range(len(x)):
        left = x[i] - W/2
        right = x[i] + W/2

        mask = (x >= left) & (x <= right)
        xw = x[mask]
        yw = y[mask]

        if len(xw) < deg + 1:
            continue

        xbar = xw.mean()
        xshift = xw - xbar

        A = vandermonde(xshift, deg)
        c = ls_qr(A, yw)

        deriv = c[1] if deg >= 1 else 0.0

        centers.append(xbar)
        derivs.append(deriv)

    return np.array(centers), np.array(derivs)


def problem_1a():
    print("==== Problem 1A (CO2 polynomial least squares) ====")
    fname = "co2_mm_mlo.csv"
    deg = int(input("Enter polynomial degree n:").strip())

    df = pd.read_csv(fname, comment='#')
    t = df["decimal date"].to_numpy()
    y = df["average"].to_numpy()

    mask = y > -90
    t = t[mask]
    y = y[mask]

    t = t - t.mean()
    t = t / t.std()

    A = vandermonde(t, deg)

    x_ne = ls_normal_eq(A, y)
    x_qr = ls_qr(A, y)

    print("Normal equations (Cholesky) coeffs:")
    for i, v in enumerate(x_ne):
        print(f"{i}, {v:.6g}")

    print("QR (Gram-Schmidt) coeffs:")
    for i, v in enumerate(x_qr):
        print(f"{i}, {v:.6g}")

    A = np.array(A, float)
    y = np.array(y, float)

    x_ne = np.array(x_ne, float)
    x_qr = np.array(x_qr, float)

    print("Residual (normal equations):", np.linalg.norm(A @ x_ne - y))
    print("Residual (QR):", np.linalg.norm(A @ x_qr - y))

    t_plot = np.linspace(t.min(), t.max(), 500)
    y_ne = eval_poly(x_ne, t_plot)
    y_qr = eval_poly(x_qr, t_plot)

    plt.plot(t, y, ".", label="data")
    plt.plot(t_plot, y_ne, label="normal equations")
    plt.plot(t_plot, y_qr, label="QR")
    plt.xlabel("scaled time")
    plt.ylabel("CO2")
    plt.title(f"CO2 least-squares fit, degree {deg}")
    plt.legend()
    plt.show()


def problem_1b():
    print("==== Problem 1B (Hilbert test) ====")
    n = int(input("Enter size of Hilbert matrix, n:").strip())

    A = hilbert(n)

    y = [0.0]*n
    m = int(input("Enter number of columns on the right side, m:").strip())

    for k in range(m):
        col = int(input(f"Enter a column (0,{n-1}):").strip())
        coef = float(input(f"Enter a coefficient for column {col}:").strip())
        for i in range(n):
            y[i] += coef*A[i][col]

    l = cholesky(A)
    x = solve_cholesky(l, y)

    print("Solution x:")
    for i, v in enumerate(x):
        print(f"{i}, {v:.6g}")

def problem_2():
    print("==== Problem 2 (DNA least-squares derivative) ====")

    data = pd.read_csv(
        "http://math.utah.edu/~palais/FvsT.txt",
        header=None,
        names=["T","F"]
    )

    T = data["T"].to_numpy()
    F = data["F"].to_numpy()

    widths = input("Enter window widths (comma separated): ").strip()
    W_list = [float(w) for w in widths.split(",")]

    plt.figure()

    for W in W_list:
        for deg in [1, 2, 3]:
            centers, derivs = local_ls_derivative(T, F, W, deg)
            plt.plot(centers, derivs, label=f"W={W}, deg={deg}")

    plt.xlabel("Temperature")
    plt.ylabel("dF/dT (least squares)")
    plt.title("DNA Melting Curve Derivative")
    plt.legend()
    plt.show()


def main():
    problem_1a()
    problem_1b()
    problem_2()

if __name__ == "__main__":
    main()