import autograd.numpy as np
import matplotlib.pyplot as plt
from gradientprosty import solver, SolverParameters
from typing import List


# === Funkcje celu ===
def f3(x):
    n = len(x)
    exponents = np.linspace(0, 1, n)
    weights = 10**6**exponents
    return np.sum(weights * x**2)


def f19(x):
    def f_ackley(x):
        return (
            -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / len(x)))
            - np.exp(np.sum(np.cos(2 * np.pi * x)) / len(x))
            + 20
            + np.e
        )

    def f_rosenbrock(x):
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    def f_weierstrass(x):
        a, b, k_max = 0.5, 3, 20
        k = np.arange(0, k_max + 1)
        sum1 = np.sum(
            [np.sum(a**k_i * np.cos(2 * np.pi * b**k_i * (x + 0.5))) for k_i in k]
        )
        sum2 = np.sum(a**k * np.cos(np.pi * b**k))
        return np.sum(sum1) - len(x) * sum2

    return 0.3 * f_ackley(x) + 0.3 * f_rosenbrock(x) + 0.4 * f_weierstrass(x)


n = 10
max_iter = 20000
tol = 1e-6

np.random.seed(13)
x0 = np.random.uniform(0, 5, size=n)

alphas_f3 = [1e-2, 1e-3, 1e-4]
alphas_f19 = [1e-3, 1e-4, 1e-5]

test_funcs = [
    ("F3", f3, alphas_f3),
    ("F19", f19, alphas_f19),
]

results = {}

for func_name, func, alphas in test_funcs:
    results[func_name] = {}
    print(f"\n=== Test funkcji {func_name} ===")

    for alpha in alphas:
        print(f"--- alpha={alpha} ---")
        params = SolverParameters(alpha=alpha, max_iter=max_iter, tol=tol)
        result = solver(func, x0, params)

        results[func_name][alpha] = result

        print(f"  Ostateczne f_opt = {result.f_opt:.6f}")
        print(f"  Iterations = {result.iterations}, Success = {result.success}")


for func_name in results:
    plt.figure()
    plt.title(f"Zbieżność gradientu prostego – {func_name}")
    plt.xlabel("Iteracja")
    plt.ylabel("Wartość funkcji")
    plt.yscale("log")

    for alpha, res in results[func_name].items():
        plt.plot(range(res.iterations), res.history, label=f"alpha={alpha}")

    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(f"{func_name}_gradient_plot.png")
    print(f"Zapisano wykres: {func_name}_gradient_plot.png")
