import numpy as np
import matplotlib.pyplot as plt


def plot_func(x, y, file_name="plot.png"):
    fig, ax = plt.subplots()
    ax.axhline(y=0, color="k")
    ax.axvline(x=0, color="k")
    ax.grid(True)
    ax.plot(x, y)
    fig.savefig(file_name, dpi=200, bbox_inches="tight")
    plt.close(fig)


# Linear Polynomial func
x = np.linspace(-5, 5, 100)
y = 0.5 * x + 1

# Quadratic func
x = np.linspace(-2, 2, 100)
y = -0.5 * 9.8 * x ** 2 + 2 * x + 1
plot_func(x, y, "plot2.png")
