# feigenbaum_tree_classic.py
# Clean, textbook-style Feigenbaum tree (logistic-map bifurcation diagram)

import numpy as np
import matplotlib.pyplot as plt

def logistic(x, r):
    return r * x * (1 - x)

def bifurcation_data(r_min=2.5, r_max=4.0, r_points=8000, n_transient=1000, n_keep=500):
    """
    Compute logistic map bifurcation data.
    Focuses on r ∈ [2.5, 4.0] — where the Feigenbaum cascade occurs.
    """
    r = np.linspace(r_min, r_max, r_points)
    x = 1e-6 * np.ones_like(r)

    # Burn-in (discard transient)
    for _ in range(n_transient):
        x = logistic(x, r)

    # Collect final points
    xs = np.empty(r_points * n_keep)
    rs = np.empty_like(xs)

    for i in range(n_keep):
        x = logistic(x, r)
        xs[i*r_points:(i+1)*r_points] = x
        rs[i*r_points:(i+1)*r_points] = r

    return rs, xs

def plot_feigenbaum_tree():
    # Generate data
    rs, xs = bifurcation_data(
        r_min=2.5, r_max=4.0,
        r_points=8000, n_transient=1000, n_keep=500
    )

    # Optional: tiny horizontal jitter to reduce overplotting artifacts
    rs += np.random.normal(scale=0.0002, size=rs.shape)

    # Plot
    plt.figure(figsize=(12, 7))
    plt.scatter(rs, xs, s=0.05, color="black", marker=".", lw=0)
    plt.xlim(2.5, 4.0)
    plt.ylim(0, 1)
    plt.title("Feigenbaum Tree — Logistic Map Bifurcation Diagram", fontsize=13)
    plt.xlabel("r")
    plt.ylabel("x (asymptotic)")
    plt.grid(False)

    # Reference landmarks
    r_marks = [
        (3.000000, "period-2"),
        (3.449489743, "period-4"),
        (3.544090, "period-8"),
        (3.564407, "period-16"),
        (3.569945672, "onset of chaos"),
        (3.828, "period-3 window")
    ]
    for r, label in r_marks:
        plt.axvline(r, ls="--", lw=0.7, alpha=0.5)
        plt.text(r, 0.05, label, rotation=90, va="bottom", ha="right", fontsize=8, alpha=0.8)

    plt.show()

if __name__ == "__main__":
    plot_feigenbaum_tree()
