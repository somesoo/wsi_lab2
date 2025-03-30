import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
from evaluacyjny import SolverParameters, SolverResult, solver
from cec2017.hybrid import f19

# Konfiguracja
dimension = 10
x0 = np.random.uniform(-100, 100, size=dimension)


def eval_func(x):
    x = np.array(x)
    if len(x) != dimension:
        raise ValueError(f"Wektor x musi mieć długość {dimension}")
    return f19(x[np.newaxis, :])[0]


# Parametry algorytmu
params = SolverParameters(
    pop_size=500, max_iter=3000, mutation_prob=0.8, mutation_strength=6.0, tol=1e-1
)

# Uruchomienie solvera
result = solver(eval_func, x0, params)

# Wyniki
print("\nNajlepsze x:", result.x_opt)
print("Najlepsza wartość funkcji:", result.f_opt)

# Wykres
plt.plot(result.history)
plt.title("Postęp optymalizacji (f19)")
plt.xlabel("Iteracja")
plt.ylabel("Najlepsze f(x)")
plt.grid(True)
plt.show()
