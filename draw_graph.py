import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_csv_and_get_iterations(filename, func_name):
    df = pd.read_csv(filename)

    # Wykres indywidualny log
    plt.figure(figsize=(10, 6))
    for col in df.columns:
        plt.plot(df[col], label=col, linewidth=1, alpha=0.7)

    plt.yscale("log")
    plt.xlabel("Iteracja")
    plt.ylabel("f(x) (log scale)")
    plt.title(f"Zbieżność ewolucyjna – {func_name}")
    plt.grid(True, which="both", ls="--")
    plt.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    plt.savefig(f"{func_name}_log_plot.png")
    print(f"Zapisano wykres: {func_name}_log_plot.png")

    # Zbieranie liczby iteracji
    iter_counts = df.notna().sum()
    iter_df = pd.DataFrame(
        {
            "Function": func_name,
            "Run": iter_counts.index,
            "Iterations": iter_counts.values,
        }
    )

    mean_history = df.mean(axis=1)

    return iter_df, mean_history


# Przetwarzanie F3 i F19
f3_iters, f3_mean = plot_csv_and_get_iterations("F3_histories.csv", "F3")
f19_iters, f19_mean = plot_csv_and_get_iterations("F19_histories.csv", "F19")

# Wspólny wykres średnich
plt.figure(figsize=(10, 6))
plt.plot(f3_mean, label="F3 - średnia", linewidth=2)
plt.plot(f19_mean, label="F19 - średnia", linewidth=2)
plt.yscale("log")
plt.xlabel("Iteracja")
plt.ylabel("Średnie f(x) (log)")
plt.title("Średnia zbieżność dla F3 i F19")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.savefig("avg_convergence_plot.png")
print("Zapisano wykres: avg_convergence_plot.png")

# Zbiorczy CSV z iteracjami
all_iters = pd.concat([f3_iters, f19_iters], ignore_index=True)

# Dodaj wiersze ze średnimi na końcu
avg_f3 = f3_iters["Iterations"].mean()
avg_f19 = f19_iters["Iterations"].mean()

avg_df = pd.DataFrame(
    [
        {"Function": "F3", "Run": "AVG", "Iterations": avg_f3},
        {"Function": "F19", "Run": "AVG", "Iterations": avg_f19},
    ]
)

all_iters = pd.concat([all_iters, avg_df], ignore_index=True)

# Zapis do pliku
all_iters.to_csv("all_iterations.csv", index=False)
print("Zapisano tabelę all_iterations.csv z uwzględnieniem średnich.")
