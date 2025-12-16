# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 18:26:55 2025

@author: Conal
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


# ----------------------------
# Mackey–Glass DDE simulator (Euler with history buffer)
# dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)
# ----------------------------
def simulate_mackey_glass(
    tau,
    beta=0.2,
    gamma=0.1,
    n=10,
    dt=0.1,
    T=2000.0,
    x0=1.2,
    warmup=500.0,
    seed=0,
):
    rng = np.random.default_rng(seed)

    steps = int(T / dt)
    warmup_steps = int(warmup / dt)

    delay_steps = int(round(tau / dt))
    if delay_steps < 1:
        raise ValueError("tau must be >= dt for this discretization")

    # History: initial condition on [-tau, 0]
    hist = np.full(delay_steps + 1, x0, dtype=float)
    x = x0

    xs = np.empty(steps, dtype=float)

    for i in range(steps):
        x_tau = hist[0]  # oldest value ~ x(t - tau)
        dx = beta * x_tau / (1.0 + (x_tau ** n)) - gamma * x
        x = x + dt * dx

        # update buffer: drop oldest, append newest
        hist[:-1] = hist[1:]
        hist[-1] = x

        xs[i] = x

    # discard warmup
    if warmup_steps >= len(xs):
        raise ValueError("warmup is too large for T")
    return xs[warmup_steps:]


# ----------------------------
# Delay embedding
# ----------------------------
def embed_delay(x, m, lag):
    N = len(x) - (m - 1) * lag
    Y = np.empty((N, m))
    for j in range(m):
        Y[:, j] = x[j * lag : j * lag + N]
    return Y



# ----------------------------
# Rosenstein-style LLE estimate
# - Find nearest neighbour of each embedded point (with Theiler window)
# - Track mean log separation growth vs time
# - Fit slope on chosen linear region
# Returns: (lambda, t_fit, mean_log_div)
# ----------------------------
def largest_lyapunov_rosenstein_fast(
    x,
    dt,
    m=4,
    lag=8,
    theiler=30,
    k_max=120,
    fit_range=(10, 50),
    ref_step=5,
):
    Y = embed_delay(x, m, lag)
    N = len(Y)

    tree = cKDTree(Y)

    # Reference points (subsampled)
    refs = np.arange(0, N, ref_step)

    nn_idx = []
    for i in refs:
        dists, inds = tree.query(Y[i], k=10)
        for j in inds:
            if abs(i - j) > theiler:
                nn_idx.append((i, j))
                break

    if len(nn_idx) < 20:
        raise RuntimeError("Too few valid neighbour pairs")

    log_div = np.zeros(k_max)
    counts = np.zeros(k_max)

    eps = 1e-12
    for i, j in nn_idx:
        k_end = min(k_max, N - max(i, j))
        if k_end < 2:
            continue
        d = np.linalg.norm(Y[i:i+k_end] - Y[j:j+k_end], axis=1)
        log_div[:k_end] += np.log(d + eps)
        counts[:k_end] += 1

    valid = counts > 0
    log_div[valid] /= counts[valid]

    k0, k1 = fit_range
    t = np.arange(k_max) * dt
    mask = valid & (np.arange(k_max) >= k0) & (np.arange(k_max) <= k1)

    coeffs = np.polyfit(t[mask], log_div[mask], 1)
    return coeffs[0], t, log_div



# ----------------------------
# Sweep tau and plot LLE vs tau
# ----------------------------
def sweep_tau_fast(taus):
    lles = []

    for tau in taus:
        x = simulate_mackey_glass(tau)
        x = (x - x.mean()) / (x.std() + 1e-12)

        # Downsample
        ds = 5
        x = x[::ds]
        dt_eff = 0.1 * ds

        lam, _, _ = largest_lyapunov_rosenstein_fast(x, dt_eff)
        lles.append(lam)

        print(f"tau={tau:5.1f}  lambda_max ≈ {lam:.4f}")

    plt.figure()
    plt.plot(taus, lles, "o-")
    plt.xlabel(r"Delay $\tau$")
    plt.ylabel(r"Largest Lyapunov exponent $\lambda_{\max}$")
    plt.title("Mackey–Glass: Largest Lyapunov Exponent vs Delay")
    plt.grid(True)
    plt.show()

    return np.array(lles)


taus = np.arange(5, 30, 1)
sweep_tau_fast(taus)
