# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 22:08:34 2025

@author: Conal
"""

import numpy as np
import matplotlib.pyplot as plt

def rossler_step(x, y, z, a=0.2, b=0.2, c=5.7):
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return dx, dy, dz

def integrate_rossler(dt=0.01, n_steps=200000, burn_in=2000, x0=(0.1, 0.0, 0.0),
                      a=0.2, b=0.2, c=5.7):
    x = np.empty(n_steps)
    y = np.empty(n_steps)
    z = np.empty(n_steps)

    x[0], y[0], z[0] = x0

    # RK4 integration
    for i in range(n_steps - 1):
        xi, yi, zi = x[i], y[i], z[i]

        k1 = np.array(rossler_step(xi, yi, zi, a, b, c))
        k2 = np.array(rossler_step(xi + 0.5*dt*k1[0], yi + 0.5*dt*k1[1], zi + 0.5*dt*k1[2], a, b, c))
        k3 = np.array(rossler_step(xi + 0.5*dt*k2[0], yi + 0.5*dt*k2[1], zi + 0.5*dt*k2[2], a, b, c))
        k4 = np.array(rossler_step(xi + dt*k3[0], yi + dt*k3[1], zi + dt*k3[2], a, b, c))

        step = (k1 + 2*k2 + 2*k3 + k4) / 6.0
        x[i+1] = xi + dt * step[0]
        y[i+1] = yi + dt * step[1]
        z[i+1] = zi + dt * step[2]

    return x[burn_in:], y[burn_in:], z[burn_in:]

if __name__ == "__main__":
    x, y, z = integrate_rossler(dt=0.01, n_steps=120000, burn_in=2000)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, linewidth=0.3)
    ax.set_title("Rössler Attractor")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    plt.show()
