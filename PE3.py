import math
import numpy as np


def f(x):
    return math.exp(x)

def romberg(a, b, max_levels=10, tol=0.001):
    I = np.zeros((max_levels, max_levels))

    exact = math.e - 1

    ea = 100
    level = 0
    h = b - a
    intervals = 1

    I[0, 0] = (h / 2) * (f(a) + f(b))

    while ea > tol and level < max_levels - 1:
        s = 0
        x = a + h / 2

        for j in range(intervals):
            s = s + f(x)
            x = x + h

        I[level + 1, 0] = 0.5 * I[level, 0] + 0.5 * h * s

        for k in range(1, level + 2):
            j = level + 1 - k
            I[j, k] = (4**k * I[j + 1, k - 1] - I[j, k - 1]) / (4**k - 1)

        ea = abs((I[0, level + 1] - I[1, level]) / I[0, level + 1]) * 100

        level = level + 1
        intervals = intervals * 2
        h = h / 2

    et = abs((exact - I[0, level]) / exact) * 100

    return I, level, ea, et


def print_romberg_table(I, level):
    print("Romberg table:")
    for i in range(level + 1):
        for j in range(level + 1 - i):
            print(I[i, j], end="\t")
        print()

def ij_from_n(n, J):
    return (n % J, n // J)

def n_from_ij(i, j, J):
    return j * J + i

def build_neighbors(J):
    Jsq = J * J
    component = [[0 for _ in range(4)] for _ in range(Jsq)]
    neighbors = [0 for _ in range(Jsq)]

    for n in range(Jsq):
        i, j = ij_from_n(n, J)
        count = 0

        if i + 1 < J:
            component[n][count] = n_from_ij(i+1, j, J)
            count += 1
        if j + 1 < J:
            component[n][count] = n_from_ij(i, j+1, J)
            count += 1
        if i - 1 >= 0:
            component[n][count] = n_from_ij(i-1, j, J)
            count += 1
        if j - 1 >= 0:
            component[n][count] = n_from_ij(i, j-1, J)
            count += 1

        neighbors[n] = count

    return component, neighbors

def rhs_function(J):
    pi = math.pi
    h = pi / (J + 1)
    f = [0.0 for _ in range(J*J)]

    k = 0
    for j in range(J):
        y = (j+1)*h
        gy = 1 - ((2/pi)*(y - pi/2))**2

        for i in range(J):
            x = (i+1)*h
            gx = 1 - ((2/pi)*(x - pi/2))**2

            f[k] = gx * gy
            k += 1

    return f

def apply_S(u, J, hsq, component, neighbors):
    Su = [0.0 for _ in range(J*J)]

    for n in range(J*J):
        s = -4.0 * u[n]
        for k in range(neighbors[n]):
            s += u[component[n][k]]
        Su[n] = s / hsq

    return Su


def jacobi_step(u, f, J, hsq, component, neighbors):
    unew = [0.0 for _ in range(J*J)]

    for n in range(J*J):
        s = 0.0
        for k in range(neighbors[n]):
            s += u[component[n][k]]
        unew[n] = -0.25 * (hsq * f[n] - s)

    return unew


def gauss_seidel_step(u, f, J, hsq, component, neighbors):
    for n in range(J*J):
        s = 0.0
        for k in range(neighbors[n]):
            s += u[component[n][k]]
        u[n] = -0.25 * (hsq * f[n] - s)


def dot(u, v):
    s = 0.0
    for n in range(len(u)):
        s += u[n] * v[n]
    return s

def norm(u):
    return math.sqrt(dot(u, u))

def eigenvector_lm(J, l, m):
    h = math.pi / (J + 1)
    e = [0.0 for _ in range(J * J)]

    for j in range(J):
        y = (j + 1) * h
        for i in range(J):
            x = (i + 1) * h
            n = n_from_ij(i, j, J)
            e[n] = math.sin(l * x) * math.sin(m * y)

    return e

def discrete_eigenvalue(J, l, m):
    h = math.pi / (J + 1)
    hsq = h * h
    return (-4.0 + 2.0 * math.cos(l * h) + 2.0 * math.cos(m * h)) / hsq

def eigenbasis_demo(J):
    h = math.pi / (J + 1)
    hsq = h * h
    component, neighbors = build_neighbors(J)

    l1 = 1
    m1 = 1
    l2 = 1
    m2 = 2

    e1 = eigenvector_lm(J, l1, m1)
    e2 = eigenvector_lm(J, l2, m2)

    print("dot(e_%d,%d, e_%d,%d) =" % (l1, m1, l2, m2), dot(e1, e2))
    print("||e_%d,%d|| =" % (l1, m1), norm(e1))
    print("||e_%d,%d|| =" % (l2, m2), norm(e2))

    Se1 = apply_S(e1, J, hsq, component, neighbors)
    lam1 = discrete_eigenvalue(J, l1, m1)

    print("lambda_%d,%d =" % (l1, m1), lam1)
    print("First few entries of S e - lambda e:")

    for n in range(min(10, J * J)):
        print(Se1[n] - lam1 * e1[n])


def expand_in_eigenbasis(v, J):
    coeff = [[0.0 for _ in range(J + 1)] for _ in range(J + 1)]

    for l in range(1, J + 1):
        for m in range(1, J + 1):
            e = eigenvector_lm(J, l, m)
            denom = dot(e, e)
            if denom != 0.0:
                coeff[l][m] = dot(v, e) / denom

    return coeff

def reconstruct_from_eigenbasis(coeff, J):
    v = [0.0 for _ in range(J * J)]

    for l in range(1, J + 1):
        for m in range(1, J + 1):
            c = coeff[l][m]
            if c != 0.0:
                e = eigenvector_lm(J, l, m)
                for n in range(J * J):
                    v[n] += c * e[n]

    return v

def solve_by_eigenbasis(rhs, J):
    coeff_rhs = expand_in_eigenbasis(rhs, J)
    coeff_u = [[0.0 for _ in range(J + 1)] for _ in range(J + 1)]

    for l in range(1, J + 1):
        for m in range(1, J + 1):
            lam = discrete_eigenvalue(J, l, m)
            if lam != 0.0:
                coeff_u[l][m] = coeff_rhs[l][m] / lam

    u = reconstruct_from_eigenbasis(coeff_u, J)
    return u, coeff_rhs

def solve_rhs_demo(J):
    h = math.pi / (J + 1)
    hsq = h * h
    component, neighbors = build_neighbors(J)

    rhs = rhs_function(J)
    u, coeff_rhs = solve_by_eigenbasis(rhs, J)
    Su = apply_S(u, J, hsq, component, neighbors)

    print("Expansion coefficients of rhs:")
    for l in range(1, J + 1):
        for m in range(1, J + 1):
            print("c_%d,%d =" % (l, m), coeff_rhs[l][m])

    print("\nFirst few entries of Su - rhs:")
    for n in range(min(10, J * J)):
        print(Su[n] - rhs[n])

def diffusion_solution(rhs, J, t):
    coeff_rhs = expand_in_eigenbasis(rhs, J)
    coeff_t = [[0.0 for _ in range(J + 1)] for _ in range(J + 1)]

    for l in range(1, J + 1):
        for m in range(1, J + 1):
            coeff_t[l][m] = coeff_rhs[l][m] * math.exp(-(l * l + m * m) * t)

    u = reconstruct_from_eigenbasis(coeff_t, J)
    return u

def diffusion_demo(J, t):
    rhs = rhs_function(J)
    u = diffusion_solution(rhs, J, t)

    print("Diffusion solution at time t =", t)
    for n in range(J * J):
        print(u[n])

def main():
    a = 0
    b = 1

    I, level, ea, et = romberg(a, b)

    print_romberg_table(I, level)
    print("\nBest approximation =", I[0, level])
    print("Approximate relative error (%) =", ea)
    print("True relative error (%) =", et)

    J = 4

    print("\nEigenbasis demo:")
    eigenbasis_demo(J)

    print("\nSolve Su = rhs by eigenbasis:")
    solve_rhs_demo(J)

    print("\nDiffusion demo:")
    diffusion_demo(J, 0.1)

if __name__ == "__main__":
    main()