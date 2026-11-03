import math


def f(rootform, x):
    if rootform == 1:
        return x*x - 2.0
    if rootform == 2:
        return x*x*x
    if rootform == 3:
        return math.copysign(abs(x)**(1.0/3.0), x)
    return 0.0


def df(rootform, x):
    if rootform == 1:
        return 2.0*x
    if rootform == 2:
        return 3.0*x*x
    if rootform == 3:
        c = math.copysign(abs(x)**(1.0/3.0), x)
        return 1.0/(3.0*c*c)
    return 0.0


def function_name(rootform):
    if rootform == 1:
        return "x^2 = 2"
    if rootform == 2:
        return "x^3 = 0"
    if rootform == 3:
        return "x^(1/3) = 0"
    return "Unknown"


def method_name(method):
    if method == 1:
        return "Bisection"
    if method == 2:
        return "Newton"
    if method == 3:
        return "Secant"
    if method == 4:
        return "False Position"
    return "Unknown"


def bisection(rootform, x0, x1, tol=1e-14, itermax=40):
    xleft = x0
    xright = x1
    fxleft = f(rootform, xleft)
    fxright = f(rootform, xright)

    if fxleft * fxright > 0:
        print("Bisection needs a bracketing interval.")
        return None

    iter = 0
    dxn = xright - xleft
    fxn = 1.0

    print("\n==== Bisection Iterations ====")
    while abs(fxn) > tol and abs(dxn) > tol and iter <= itermax:
        xnew = 0.5 * (xleft + xright)
        fxnew = f(rootform, xnew)

        if fxnew * fxleft < 0:
            dxn = xright - xnew
            xright = xnew
            fxright = fxnew
        else:
            dxn = xleft - xnew
            xleft = xnew
            fxleft = fxnew

        fxn = fxnew
        iter += 1
        print(f"{iter}: [{xleft:.12g},{xright:.12g}]")

    return 0.5 * (xleft + xright)


def newton(rootform, x0, tol=1e-14, itermax=40):
    xn = x0
    iter = 0
    dxn = 1.0
    fxn = f(rootform, xn)

    print("\n==== Newton Iterations ====")
    while abs(fxn) > tol and abs(dxn) > tol and iter <= itermax:
        fxn = f(rootform, xn)
        dfxn = df(rootform, xn)
        dxn = -fxn / dfxn
        xn = xn + dxn

        iter += 1
        print(f"{iter}: {xn:.12g}")

        fxn = f(rootform, xn)

    return xn


def secant(rootform, x0, x1, tol=1e-14, itermax=40):
    xm = x0
    xn = x1
    fxm = f(rootform, xm)
    fxn = f(rootform, xn)

    iter = 0
    dxn = xn - xm

    print("\n==== Secant Iterations ====")
    while abs(fxn) > tol and abs(dxn) > tol and iter <= itermax:
        xtemp = (xn * fxm - xm * fxn) / (fxm - fxn)
        dxn = xtemp - xn

        xm = xn
        fxm = fxn
        xn = xtemp
        fxn = f(rootform, xn)

        iter += 1
        print(f"{iter}: {xn:.12g}")

    return xn


def false_position(rootform, x0, x1, tol=1e-14, itermax=40):
    xleft = x0
    xright = x1
    fxleft = f(rootform, xleft)
    fxright = f(rootform, xright)

    if fxleft * fxright > 0:
        print("False position needs a bracketing interval.")
        return None

    iter = 0
    dxn = xright - xleft
    fxn = 1.0
    xnew = xleft

    print("\n==== False Position Iterations ====")
    while abs(fxn) > tol and abs(dxn) > tol and iter <= itermax:
        xnew = (xleft * fxright - xright * fxleft) / (fxright - fxleft)
        fxnew = f(rootform, xnew)

        if fxnew * fxleft < 0:
            dxn = xright - xnew
            xright = xnew
            fxright = fxnew
        else:
            dxn = xleft - xnew
            xleft = xnew
            fxleft = fxnew

        fxn = fxnew
        iter += 1
        print(f"{iter}: [{xleft:.12g},{xright:.12g}]")

    return xnew


def main():
    Bisection = 1
    Newton = 2
    Secant = 3
    False_Position = 4

    print("==== Root-Finding Methods ====")
    print("Choose: 1 (x^2=2) 2 (x^3=0) 3 (x^(1/3)=0)")
    rootform = int(input().strip())

    print("\nChoose: 1 (Bisection) 2 (Newton) 3 (Secant) 4 (False Position)")
    method = int(input().strip())

    print("\nEnter initial approximation x0 (the lesser, leftmost if bracketing needed):")
    xzero = float(input().strip())

    xone = None
    if method == Bisection or method == Secant or method == False_Position:
        print("\nEnter second initial approximation x1:")
        xone = float(input().strip())

    print("\n==== Problem Setup ====")
    print("Function:", function_name(rootform))
    print("Method:", method_name(method))
    print("Tolerance =", 1e-14)
    print("Maximum iterations =", 40)

    if method == Newton:
        answer = newton(rootform, xzero)

    if method == Secant:
        answer = secant(rootform, xzero, xone)

    if method == False_Position:
        answer = false_position(rootform, xzero, xone)

    if method == Bisection:
        answer = bisection(rootform, xzero, xone)

    print("\n==== Final Result ====")
    print("Approximation =", answer)
    print("f(x) =", f(rootform, answer))


if __name__ == "__main__":
    main()
