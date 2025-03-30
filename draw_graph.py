import matplotlib.pyplot as plt
import pandas as pd

def plot_csv_and_get_iterations(filename, func_name):
    df = pd.read_csv(filename)

    # Rysowanie wykresu
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
    plot_filename = f"{func_name}_log_plot.png"
    plt.savefig(plot_filename)
    print(f"Zapisano wykres: {plot_filename}")

    # Liczba iteracji (czyli nie-NaN wartości)
    iter_counts = df.notna().sum()
    iter_df = pd.DataFrame({
        "Function": func_name,
        "Run": iter_counts.index,
        "Iterations": iter_counts.values
    })
    return iter_df

# Przetwarzanie obu funkcji
f3_iters = plot_csv_and_get_iterations("F3_histories.csv", "F3")
f19_iters = plot_csv_and_get_iterations("F19_histories.csv", "F19")

# Łączymy do jednej tabeli i zapisujemy
all_iters = pd.concat([f3_iters, f19_iters], ignore_index=True)
all_iters.to_csv("all_iterations.csv", index=False)
print("\nZapisano zbiorczą tabelę iteracji do: all_iterations.csv")
