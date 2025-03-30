import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
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
        f_val = eval_func(x)




        history.append(f_val)


    
    end_time = time.time()
    total_time = end_time - start_time
        
    return SolverResult(
        x_opt=x,
        f_opt=f_opt,
        iterations=iterations,
        success=success,
        history=history
    )
