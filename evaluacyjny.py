import autograd.numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from autograd import grad
import time
from dataclasses import dataclass
from typing import Callable, Sequence, List
import cec2017


dimension = 10
func_num = 3  # F3
benchmark = cec2017.CEC2017(dimension)
benchmark.set_function(func_num)


# Funkcja celu (kompatybilna z autograd)
def eval_func(x):
    return benchmark.evaluate(x)

# Gradient funkcji celu
grad_func = grad(eval_func)


@dataclass
class SolverParameters:
    alpha: float
    max_iter: int
    tol: float

@dataclass
class SolverResult:
    x_opt: np.ndarray
    f_opt: float
    iterations: int
    success: bool
    history: List[float]

def solver(
    eval_func: Callable[[np.ndarray], float],
    x0: Sequence[float],
    params: SolverParameters
) -> SolverResult:
    
    x = np.array(x0, dtype=float)
    history = []
    success = False
    start_time = time.time()
    
    for i in range(params.max_iter):
        f_val = eval_func(x)
        history.append(f_val)

        g = grad_func(x)
        x_next = x - params.alpha * g

        if np.linalg.norm(x_next - x) < params.tol:
            success = True
            x = x_next
            break

        x = x_next
        if not np.isfinite(f_val):
            break
    
    end_time = time.time()
    total_time = end_time - start_time

    f_opt = eval_func(x)
    iterations = i + 1

    print(f"Solver zakończony w {iterations} iteracjach (czas: {total_time:.4f}s). "
          f"f_opt={f_opt:.4e}, ||x||={np.linalg.norm(x):.3f}, success={success}")
        
    return SolverResult(
        x_opt=x,
        f_opt=f_opt,
        iterations=iterations,
        success=success,
        history=history
    )

if __name__ == "__main__":

    # Parametry ogólne
    n = 10
    max_iter = 20000
    tol = 1e-6

    np.random.seed(11)
    x0 = np.random.uniform(-100, 100, size=n)

    alphas = [1e-3, 1e-4, 1e-5]

    # results[func_name][alpha] = SolverResult
    results = {}

    for alpha in alphas:
        print(f"\n=== alpha = {alpha} ===")
        params = SolverParameters(alpha=alpha, max_iter=max_iter, tol=tol)
        result = solver(eval_func, x0, params)
        results[alpha] = result

    plt.figure()
    plt.title(f"Przebieg wartości F{func_num}")
    plt.xlabel("Iteracja")
    plt.ylabel("Wartość funkcji")

    for alpha in alphas:
        res = results[alpha]
        plt.plot(range(res.iterations), res.history, label=f"alpha={alpha}")

    plt.legend()
    plt.grid(True)
    plt.show()