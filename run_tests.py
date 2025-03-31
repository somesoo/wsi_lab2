import numpy as np
import csv
from evaluacyjny import SolverParameters, solver
from cec2017.simple import f3
from cec2017.hybrid import f19

# Config
dimension = 10
repeats = 10

params = SolverParameters(
    pop_size=500, max_iter=3000, mutation_prob=0.8, mutation_strength=6.0, tol=1e-1
)


def wrap_cec_func(cec_func):
    def eval_func(x):
        x = np.array(x)
        return cec_func(x[np.newaxis, :])[0]

    return eval_func


def run_and_save_to_csv(eval_func, func_name):
    all_histories = []

    for i in range(repeats):
        x0 = np.random.uniform(-100, 100, size=dimension)
        result = solver(eval_func, x0, params)
        all_histories.append(result.history)

    # Transpose (iterations x runs)
    max_len = max(len(h) for h in all_histories)
    padded = [h + [np.nan] * (max_len - len(h)) for h in all_histories]
    padded = list(map(list, zip(*padded)))  # transpose to iteration-wise

    # Save to CSV
    csv_path = f"{func_name}_histories.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"Run_{i+1}" for i in range(repeats)]
        writer.writerow(header)
        writer.writerows(padded)

    print(f"Saved {func_name} histories to {csv_path}")


# Run and save
f3_eval = wrap_cec_func(f3)
f19_eval = wrap_cec_func(f19)

run_and_save_to_csv(f3_eval, "F3")
run_and_save_to_csv(f19_eval, "F19")
