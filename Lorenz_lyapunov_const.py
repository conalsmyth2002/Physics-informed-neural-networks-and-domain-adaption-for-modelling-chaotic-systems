# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 20:18:54 2025

@author: Conal
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


# ----------------------------
# Mackey–Glass DDE simulator (Euler with history buffer)
# dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)
# ----------------------------
def simulate_lorenz63(
    rho,
    sigma=10.0,
    beta=8/3,
    dt=0.01,
    T=100.0,
    warmup=20.0,
    x0=(1.0, 1.0, 1.0),
):
    steps = int(T / dt)
    warmup_steps = int(warmup / dt)

    x, y, z = x0
    xs = np.empty(steps)

    def f(x, y, z):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return dx, dy, dz

    for i in range(steps):
        k1 = f(x, y, z)
        k2 = f(x + dt*k1[0]/2, y + dt*k1[1]/2, z + dt*k1[2]/2)
        k3 = f(x + dt*k2[0]/2, y + dt*k2[1]/2, z + dt*k2[2]/2)
        k4 = f(x + dt*k3[0], y + dt*k3[1], z + dt*k3[2])

        x += dt*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6
        y += dt*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6
        z += dt*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6

        xs[i] = x  # scalar observable

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
def sweep_rho_lorenz(rhos):
    lles = []

    for rho in rhos:
        x = simulate_lorenz63(rho)
        x = (x - x.mean()) / (x.std() + 1e-12)

        # Downsample
        ds = 5
        dt = 0.01
        x = x[::ds]
        dt_eff = dt * ds

        lam, _, _ = largest_lyapunov_rosenstein_fast(x, dt_eff)
        lles.append(lam)

        print(f"rho={rho:5.1f}  lambda_max ≈ {lam:.4f}")

    plt.figure()
    plt.plot(rhos, lles, "o-")
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"Largest Lyapunov exponent $\lambda_{\max}$")
    plt.title("Lorenz–63: Largest Lyapunov Exponent vs $\\rho$")
    plt.grid(True)
    plt.show()

    return np.array(lles)


rhos = np.linspace(0, 50, 26)
sweep_rho_lorenz(rhos)

