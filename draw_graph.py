import matplotlib.pyplot as plt
import pandas as pd

def plot_csv(filename, func_name):
    df = pd.read_csv(filename)
    
    plt.figure(figsize=(10, 6))
    for col in df.columns:
        plt.plot(df[col], label=col, linewidth=1, alpha=0.7)
    
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("f(x) (log scale)")
    plt.title(f"Zbieżność ewolucyjna - {func_name}")
    plt.grid(True, which="both", ls="--")
    plt.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    plt.savefig(f"{func_name}_log_plot.png")
    print(f"Saved: {func_name}_log_plot.png")

# Run plots
plot_csv("F3_histories.csv", "F3")
plot_csv("F19_histories.csv", "F19")
