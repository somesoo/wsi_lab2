import numpy as np
import time
from dataclasses import dataclass
from typing import Callable, Sequence, List
from cec2017.functions import f3, f19


@dataclass
class SolverParameters:
    pop_size: int
    max_iter: int
    mutation_prob: float
    mutation_strength: float
    tol: float


@dataclass
class SolverResult:
    x_opt: np.ndarray
    f_opt: float
    iterations: int
    success: bool
    history: List[float]


def solver(
    eval_func: Callable[[np.ndarray], float], x0: np.ndarray, params: SolverParameters
) -> SolverResult:

    dimension = len(x0)
    pop = np.random.uniform(-100, 100, size=(params.pop_size, dimension))
    history = []
    success = False
    start_time = time.time()

    global_best = None
    global_best_fit = float("inf")

    improve_c = 0
    max_improve_c = 300

    for i in range(params.max_iter):
        fitness = np.array([eval_func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_ind = pop[best_idx]
        best_fit = fitness[best_idx]
        history.append(best_fit)

        if best_fit < global_best_fit:
            global_best_fit = best_fit
            global_best = best_ind.copy()
            improve_c = 0
        else:
            improve_c += 1

        if improve_c == max_improve_c:
            success = True
            print(
                f"Brak poprawy przez {max_improve_c} iteracji. Koniec dalszych obliczeń."
            )
            break

        mut_strength = params.mutation_strength * (1 - i / params.max_iter)

        new_pop = []

        # Selekcja + mutacja
        for _ in range(params.pop_size):
            parent = pop[np.random.randint(params.pop_size)].copy()
            if np.random.rand() < params.mutation_prob:
                mutation = np.random.normal(0, mut_strength, size=dimension)
                parent += mutation
            new_pop.append(parent)

        new_pop[0] = global_best.copy()

        pop = np.array(new_pop)

    end_time = time.time()
    print(f"Zakończono po {i+1} iteracjach, f_opt = {best_fit:.4e}")

    return SolverResult(
        x_opt=best_ind,
        f_opt=best_fit,
        iterations=i + 1,
        success=success,
        history=history,
    )
