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
    eval_func: Callable[[np.ndarray], float],
    x0: np.ndarray,
    params: SolverParameters
) -> SolverResult:
    

    dimension = len(x0)
    pop = np.random.uniform(-100,100, size=(params.pop_size, dimension))
    history = []
    success = False
    start_time = time.time()
    
    for i in range(params.max_iter):
        fitness = np.array([eval_func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_ind = pop[best_idx]
        best_fit = fitness[best_idx]
        history.append(best_fit)

        new_pop = []

        # Selekcja + mutacja
        for _ in range(params.pop_size):
            parent = pop[np.random.randint(params.pop_size)].copy()
            if np.random.rand() < params.mutation_prob:
                mutation = np.random.normal(0, params.mutation_strength, size=dimension)
                parent += mutation
            new_pop.append(parent)

        pop = np.array(new_pop)

        if i > 0 and abs(history[-2] - best_fit) < params.tol:
            success = True
            break

    end_time = time.time()
    print(f"Zakończono po {i+1} iteracjach, f_opt = {best_fit:.4e}")
        
    return SolverResult(
        x_opt=best_ind,
        f_opt=best_fit,
        iterations=i+1,
        success=success,
        history=history
    )
