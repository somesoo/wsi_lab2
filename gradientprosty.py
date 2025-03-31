import autograd.numpy as np
from autograd import grad
import time
from dataclasses import dataclass
from typing import Callable, Sequence, List


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
    params: SolverParameters,
) -> SolverResult:

    x = np.array(x0, dtype=float)
    grad_func = grad(eval_func)
    history = []
    success = False

    start_time = time.time()

    for i in range(params.max_iter):
        f_val = eval_func(x)
        history.append(f_val)

        g = grad_func(x)

        x_next = x - params.alpha * g
        x_next = np.clip(x_next, -100, 100)

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

    print(
        f"Solver zakonczony w {iterations} iteracjach (czas: {total_time:.4f}s). "
        f"f_opt={f_opt:.4e}, ||x||={np.linalg.norm(x):.3f}, success={success}"
    )

    return SolverResult(
        x_opt=x, f_opt=f_opt, iterations=iterations, success=success, history=history
    )
