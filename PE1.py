import numpy as np

def power_sum(p, n):
    sum = 0
    for k in range(n + 1):
        sum += k**p
    return sum

def build_interpolation_system(p, n_values):
    degree = p+1
    num_unknowns = degree + 1

    A = np.zeros((num_unknowns, num_unknowns))
    b = np.zeros(num_unknowns)
    
    for i, n in enumerate(n_values):
        for j in range(num_unknowns):
            A[i,j] = n**j

        b[i] = power_sum(p,n)

    return A,b

def eval_poly(c, x):
    val = 0.0
    power = 1.0
    for coeff in c:
        val += coeff*power
        power *= x
    return val
    
def gaussian_elimination(A, b, pivoting):
    n= len(A)

    augmented = np.column_stack((A, b))
    augmented = augmented.astype(float)

    if pivoting == "scaled":
        s = np.max(np.abs(A), axis=1).astype(float)
        s[s == 0] = 1.0

    for i in range(n):

        pivot_row = i

        if pivoting == "partial":
            max_val = abs(augmented[i][i])
            for k in range(i+1, n):
                if abs(augmented[k][i]) > max_val:
                    max_val = abs(augmented[k][i])
                    pivot_row = k

        if pivoting == "scaled":
            max_ratio = abs(augmented[i][i]) / s[i]
            for k in range(i+1, n):
                ratio = abs(augmented[k][i]) / s[k]
                if ratio > max_ratio:
                    max_ratio = ratio
                    pivot_row = k

        if pivot_row != i:
            augmented[i], augmented[pivot_row] = augmented[pivot_row].copy(), augmented[i].copy()
            if pivoting == "scaled":
                s[i], s[pivot_row] = s[pivot_row], s[i]

        if abs(augmented[i][i]) < 1e-10:
            return "System has no unique solution"
        
        pivot = augmented[i][i]
        augmented[i] = augmented[i] / pivot

        for j in range(i+1, n):
            factor = augmented[j][i]
            augmented[j] = augmented[j] - factor*augmented[i]

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i]= augmented[i][-1]
        for j in range(i+1, n):
            x[i] = x[i] - augmented[i][j] * x[j]

    return x

def run_p(p, sets, pivoting):
    for n_values in sets:
        A,b = build_interpolation_system(p, n_values)
        c = gaussian_elimination(A, b, pivoting)
        print("n_values =", n_values, "coeffs =", c)

def divided_difference_table(x,y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)

    table = np.zeros((n,n), dtype=float)
    table[:,0] = y

    for j in range(1, n):
        for i in range(n-j):
            denom = x[i+j] - x[i]
            table[i,j] = (table[i+1, j-1] - table[i, j-1]) / denom
    
    return table

def newton_coeffs(table):
    return table[0,:]

def newton(a, x_nodes, x):
    x_nodes = np.array(x_nodes, dtype=float)
    a = np.array(a, dtype=float)

    n = len(x_nodes)
    result = a[0]
    term = 1.0
    for k in range (1,n):
        term *= (x - x_nodes[k-1])
        result += a[k] * term
    return result

def main():
    
    # Question 1
    print('\n==== Question 1: Interpolation System ====')

    p = 1
    n_values = [0, 1, 2]

    A, b = build_interpolation_system(p, n_values)

    print("A =", A)
    print("b =", b)


    # Question 2
    print('\n==== Question 2: Gaussian Elimination ====')
    print("Solution (partial pivoting):", gaussian_elimination(A, b, "partial"))
    print("Solution (scaled pivoting):", gaussian_elimination(A, b, "scaled"))

    # Question 3
    print("\n===== Question 3: Check =====")

    sets_p1 = [[0,1,2],[0,2,4]]
    for s in sets_p1:
        A,b = build_interpolation_system(1, s)
        c = gaussian_elimination(A,b,"partial")
        print("points:", s, "coeffs:", c)

    sets_p2 = [[0,1,2,3],[0,2,4,6]]
    for s in sets_p2:
        A,b = build_interpolation_system(2, s)
        c = gaussian_elimination(A,b,"partial")
        print("points:", s, "coeffs:", c)

    # Question 4
    print("\n===== Question 4: Bernoulli =====")
    p = 10
    n_values = list(range(12))

    A,b = build_interpolation_system(p, n_values)
    c = gaussian_elimination(A,b,"partial")

    n = 1000
    print("Sum of 10th powers to 1000:")
    print("Polynomial:", eval_poly(c,n))
    print("Direct sum:", power_sum(p,n))

    # Question 5
    print("\n===== Question 5: Divided Difference Table =====")

    p = 1
    n_values = [0, 1, 2]

    y_values = [power_sum(p, n) for n in n_values]

    table = divided_difference_table(n_values, y_values)

    print("points:", n_values)
    print("y values:", y_values)
    print("Divided difference table:")
    print(table)
    print("Newton coefficients:", table[0])

    # Question 6
    print("\n===== Question 6: Polynomial Evaluation =====")
    p = 1
    n_values = [0,1,2]
    A, b = build_interpolation_system(p, n_values)
    c = gaussian_elimination(A, b, "partial")
    x = 5
    print("Coefficient from P(5):", eval_poly(c, x))

    y_values = [power_sum(p, n) for n in n_values]
    table = divided_difference_table(n_values, y_values)
    a = table[0]

    print("Newton form P(5):", newton(a, n_values, x))
    print("True sum:", power_sum(p, x))

if __name__ == "__main__":
    main()