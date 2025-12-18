# -*- coding: utf-8 -*-

#!!!!! MUST READ !!!!!
#
#       If libraries are missing use terminal command:
#       pip install scipy
# or    pip install torch   etc...
#
# Current python version is 3.11.9


"""
@author: Conal Smyth

Inverse Lorenz-63 from x-only noisy data
v2.3_abrupt_rho:

- True Lorenz system has an abrupt change in rho at t_jump
- rho(t) = rho1_true for t < t_jump, rho2_true for t >= t_jump
- Inverse model learns sigma, rho1, rho2, beta
- x-only noisy observations, multi-step loss, Takens embedding, weak priors
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------
# Device & seeds
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

# ----------------------------
# Ground-truth Lorenz parameters (with abrupt rho)
# ----------------------------
sigma_true = 10.0
rho1_true  = 20.0        # below chaotic regime
rho2_true  = 28.0        # classical chaotic Lorenz
beta_true  = 8.0 / 3.0
t_jump     = 20.0        # time of abrupt change in rho

# ----------------------------
# Simulation setup
# ----------------------------
t0, t1, dt = 0.0, 45.0, 0.01
t_full = np.arange(t0, t1 + dt, dt)   # [N]
N = len(t_full)

# Index at which jump happens
jump_idx = int(np.searchsorted(t_full, t_jump))

train_frac = 2.0 / 3.0
N_train = int(train_frac * N)
t_train_end = t_full[N_train]

# Number of trajectories
K = 3

# ----------------------------
# Generate synthetic Lorenz-63 data (multiple trajectories)
# with abrupt rho(t)
# ----------------------------
def lorenz63_rhs_abrupt_rho(t, s, sigma, beta):
    rho = rho1_true if t < t_jump else rho2_true
    x, y, z = s
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    ]

X_true_list = []
u0_true_list = []

for k in range(K):
    u0_true = np.random.uniform(-15.0, 15.0, size=3)
    u0_true_list.append(u0_true)

    sol = solve_ivp(
        lorenz63_rhs_abrupt_rho,
        (t0, t1),
        u0_true,
        t_eval=t_full,
        args=(sigma_true, beta_true),
        rtol=1e-9,
        atol=1e-12,
    )
    X_true_list.append(sol.y.T)   # [N,3]

X_true = np.stack(X_true_list, axis=0)    # [K,N,3]
x_true = X_true[..., 0]                  # [K,N]

# ----------------------------
# x-only noisy observations + smoothing & derivative
# ----------------------------
rng = np.random.default_rng(0)
noise_std = 0.5

x_obs_full = x_true + rng.normal(0.0, noise_std, size=x_true.shape)  # [K,N]

win_len = 31  # must be odd
poly = 3

x_smooth_full = np.empty_like(x_obs_full)
xdot_smooth_full = np.empty_like(x_obs_full)

for k in range(K):
    x_smooth_full[k] = savgol_filter(x_obs_full[k], win_len, poly)
    xdot_smooth_full[k] = savgol_filter(
        x_obs_full[k],
        win_len,
        poly,
        deriv=1,
        delta=dt
    )

# Sparse "measurement times" for a direct data term
obs_frac = 0.25
obs_idx_list = []
for k in range(K):
    all_idx = np.arange(N)
    n_obs = int(obs_frac * N)
    obs_idx = np.sort(rng.choice(all_idx, size=n_obs, replace=False))
    obs_idx_list.append(obs_idx)

# ----------------------------
# Convert to tensors
# ----------------------------
t_raw = torch.tensor(t_full, dtype=torch.float32, device=device)          # [N]
X_true_t = torch.tensor(X_true, dtype=torch.float32, device=device)       # [K,N,3]

x_obs_t_full   = torch.tensor(x_obs_full, dtype=torch.float32, device=device)      # [K,N]
x_smooth_t     = torch.tensor(x_smooth_full, dtype=torch.float32, device=device)   # [K,N]
xdot_smooth_t  = torch.tensor(xdot_smooth_full, dtype=torch.float32, device=device)# [K,N]

train_mask = torch.zeros(N, dtype=torch.bool, device=device)
train_mask[:N_train] = True
val_mask = ~train_mask

# Precompute training obs indices (for sparse data term)
obs_idx_train_list = []
for k in range(K):
    obs_idx = obs_idx_list[k]
    obs_idx_train = obs_idx[obs_idx < N_train]
    obs_idx_train_list.append(
        torch.tensor(obs_idx_train, device=device, dtype=torch.long)
    )

# ----------------------------
# Multi-step indices (dense data term)
# ----------------------------
H = 10               # prediction horizon (in steps)
ms_stride = 5        # stride between multi-step windows

base_idx = np.arange(0, N_train - H, ms_stride)
base_idx_t = torch.tensor(base_idx, device=device, dtype=torch.long)      # [M]
steps = torch.arange(0, H + 1, device=device, dtype=torch.long)          # [H+1]
ms_idx_matrix = base_idx_t[:, None] + steps[None, :]                     # [M,H+1]

# ----------------------------
# Takens embedding indices
# ----------------------------
m_embed = 4   # embedding dimension
d_embed = 5   # delay in steps

embed_start = (m_embed - 1) * d_embed
embed_base_idx = np.arange(embed_start, N_train)  # valid times
embed_base_idx_t = torch.tensor(embed_base_idx, device=device, dtype=torch.long)   # [E]
embed_steps = torch.arange(0, m_embed, device=device, dtype=torch.long) * d_embed  # [m]
embed_idx_matrix = embed_base_idx_t[:, None] - embed_steps[None, :]                # [E,m]

# ----------------------------
# Lorenz RHS + RK4 integration (vectorised) with abrupt rho(t)
# ----------------------------
def F(u, sigma_t, rho_t, beta_t):
    """
    u: [K,3] or [B,3]
    sigma_t, rho_t, beta_t: scalar tensors
    """
    x = u[..., 0:1]
    y = u[..., 1:2]
    z = u[..., 2:3]
    return torch.cat([
        sigma_t * (y - x),
        x * (rho_t - z) - y,
        x * y - beta_t * z
    ], dim=-1)

def rk4_step(u, dt, sigma_t, rho_t, beta_t):
    k1 = F(u, sigma_t, rho_t, beta_t)
    k2 = F(u + 0.5 * dt * k1, sigma_t, rho_t, beta_t)
    k3 = F(u + 0.5 * dt * k2, sigma_t, rho_t, beta_t)
    k4 = F(u +       dt * k3, sigma_t, rho_t, beta_t)
    return u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def integrate_trajectories_abrupt_rho(u0_batch, log_sigma,
                                      log_rho1, log_rho2, log_beta,
                                      t_raw, jump_idx, noise_std=0.0):
    """
    u0_batch: [K,3]
    rho(t) = exp(log_rho1) for n < jump_idx, exp(log_rho2) for n >= jump_idx
    returns U_pred: [N,K,3]
    """
    N = t_raw.shape[0]
    dt = t_raw[1] - t_raw[0]

    sigma = torch.exp(log_sigma)
    rho1  = torch.exp(log_rho1)
    rho2  = torch.exp(log_rho2)
    beta  = torch.exp(log_beta)

    u = u0_batch.view(-1, 3)  # [K,3]
    out = [u]

    for n in range(N - 1):
        rho_t = rho1 if n < jump_idx else rho2
        u = rk4_step(u, dt, sigma, rho_t, beta)
        if noise_std > 0.0:
            u = u + noise_std * torch.randn_like(u)
        out.append(u)

    U = torch.stack(out, dim=0)   # [N,K,3]
    return U

# ----------------------------
# Weak chaos prior (abrupt rho)
# ----------------------------
def lorenz_jacobian_rho(u, sigma, rho_vec, beta):
    """
    u: [B,3]
    rho_vec: [B] (rho at each sampled point)
    sigma, beta: scalars
    """
    x = u[:, 0]
    y = u[:, 1]
    z = u[:, 2]
    B = u.shape[0]
    J = torch.zeros(B, 3, 3, device=u.device)

    J[:, 0, 0] = -sigma
    J[:, 0, 1] =  sigma

    J[:, 1, 0] = rho_vec - z
    J[:, 1, 1] = -1.0
    J[:, 1, 2] = -x

    J[:, 2, 0] = y
    J[:, 2, 1] = x
    J[:, 2, 2] = -beta
    return J

def weak_chaos_loss_abrupt_rho(U_train, rho_t_vec, sigma, beta,
                               lam_low=0.0, num_samples=128):
    """
    Very mild prior: discourages strongly negative λ_max.
    U_train: [N_train,K,3]
    rho_t_vec: [N], rho(t) over full series
    """
    N_train_, K_, _ = U_train.shape
    U_flat = U_train.reshape(-1, 3)   # [N_train*K,3]

    # rho_train: [N_train]
    rho_train = rho_t_vec[:N_train]
    # Expand per-trajectory: [N_train*K]
    rho_flat = rho_train.repeat_interleave(K)

    B_total = U_flat.shape[0]
    if B_total <= 2:
        return (torch.tensor(0.0, device=U_train.device),
                torch.tensor(0.0, device=U_train.device))

    low = B_total // 4
    high = 3 * B_total // 4
    if high <= low:
        low = 0
        high = B_total

    sample_size = min(num_samples, B_total)
    idx = torch.randint(low=low, high=high, size=(sample_size,),
                        device=U_train.device)

    u_sample = U_flat[idx]          # [B_s,3]
    rho_sample = rho_flat[idx]      # [B_s]

    J = lorenz_jacobian_rho(u_sample, sigma, rho_sample, beta)  # [B_s,3,3]
    eigvals = torch.linalg.eigvals(J)                           # [B_s,3]
    max_real = eigvals.real.max(dim=-1).values                  # [B_s]
    lam = max_real.mean()

    chaos_reg = torch.relu(lam_low - lam) ** 2
    return chaos_reg, lam

# ----------------------------
# Learnable parameters (sigma, rho1, rho2, beta)
# ----------------------------
log_sigma = nn.Parameter(torch.log(torch.tensor(10.0, device=device)))

log_rho1 = nn.Parameter(torch.log(torch.tensor(18.0, device=device)))
log_rho2 = nn.Parameter(torch.log(torch.tensor(30.0, device=device)))

log_beta = nn.Parameter(torch.log(torch.tensor(2.0, device=device)))

u0_init_np = np.array(u0_true_list, dtype=np.float32)    # (K,3)
u0_init = torch.tensor(u0_init_np, dtype=torch.float32, device=device)
u0_init = u0_init + 0.5 * torch.randn_like(u0_init)      
u0_param = nn.Parameter(u0_init)

params = [log_sigma, log_rho1, log_rho2, log_beta, u0_param]

# ----------------------------
# Loss weights and components
# ----------------------------
mse   = nn.MSELoss()
huber = nn.HuberLoss(delta=0.1)

lambda_data_sparse   = 1.0
lambda_multistep     = 3.0
lambda_takens        = 2.0
lambda_param         = 1e-4
lambda_energy        = 1e-3
lambda_chaos         = 0.5
lambda_u0            = 1e-3

total_adam_steps = 1000
process_noise_std = 0.01

# Weak priors on parameters
mu_sigma = sigma_true
mu_rho1  = rho1_true
mu_rho2  = rho2_true
mu_beta  = beta_true

tau_sigma = 10.0
tau_rho1  = 10.0
tau_rho2  = 10.0
tau_beta  = 5.0

# Energy bounds
E_min, E_max = 50.0, 3000.0

# ----------------------------
# Loss function
# ----------------------------
def pinn_loss_abrupt_rho(log_sigma, log_rho1, log_rho2, log_beta,
                         noise_std):
    sigma = torch.exp(log_sigma)
    rho1  = torch.exp(log_rho1)
    rho2  = torch.exp(log_rho2)
    beta  = torch.exp(log_beta)

    # Integrate
    U_pred = integrate_trajectories_abrupt_rho(
        u0_param, log_sigma, log_rho1, log_rho2, log_beta,
        t_raw, jump_idx, noise_std=noise_std
    )  # [N,K,3]

    x_pred = U_pred[..., 0]  # [N,K]
    y_pred = U_pred[..., 1]  # [N,K]

    # Build rho(t) vector for all times [N]
    rho_t_vec = torch.empty(N, device=device)
    rho_t_vec[:jump_idx] = rho1
    rho_t_vec[jump_idx:] = rho2

    # ------------------------
    # (1) Sparse data term (x-only)
    # ------------------------
    sparse_x_pred = []
    sparse_x_obs = []

    for k, idx_t in enumerate(obs_idx_train_list):
        if idx_t.numel() == 0:
            continue
        sparse_x_pred.append(x_pred[idx_t, k])
        sparse_x_obs.append(x_obs_t_full[k, idx_t])

    if len(sparse_x_pred) > 0:
        sxp = torch.cat(sparse_x_pred)
        sxo = torch.cat(sparse_x_obs)
        data_sparse = huber(sxp, sxo)
    else:
        data_sparse = torch.tensor(0.0, device=device)

    # ------------------------
    # (2) Multi-step dense prediction loss
    # ------------------------
    ms_losses = []
    for k in range(K):
        pred_win_k = x_pred[ms_idx_matrix, k]          # [M,H+1]
        obs_win_k  = x_obs_t_full[k, ms_idx_matrix]    # [M,H+1]
        ms_losses.append(mse(pred_win_k, obs_win_k))
    multistep_loss = torch.stack(ms_losses).mean()

    # ------------------------
    # (3) Takens embedding loss
    # ------------------------
    takens_losses = []
    for k in range(K):
        pred_emb_k = x_pred[embed_idx_matrix, k]        # [E,m]
        obs_emb_k  = x_smooth_t[k, embed_idx_matrix]    # [E,m]
        takens_losses.append(mse(pred_emb_k, obs_emb_k))
    takens_loss = torch.stack(takens_losses).mean()

    # ------------------------
    # (4) Parameter prior
    # ------------------------
    param_reg = (
        ((sigma - mu_sigma) / tau_sigma) ** 2 +
        ((rho1  - mu_rho1)  / tau_rho1)  ** 2 +
        ((rho2  - mu_rho2)  / tau_rho2)  ** 2 +
        ((beta  - mu_beta)  / tau_beta)  ** 2
    )

    # ------------------------
    # (5) Energy regularisation
    # ------------------------
    U_train = U_pred[:N_train]  # [N_train,K,3]
    energy = (U_train**2).sum(dim=-1).mean()
    energy_reg = (
        torch.relu(E_min - energy) ** 2 +
        torch.relu(energy - E_max) ** 2
    )

    # ------------------------
    # (6) Weak chaos prior with time-varying rho
    # ------------------------
    chaos_reg, lam_est = weak_chaos_loss_abrupt_rho(
        U_train, rho_t_vec, sigma, beta
    )

    # ------------------------
    # (7) u0 regularisation
    # ------------------------
    u0_reg = (u0_param**2).mean()

    # ------------------------
    # Total
    # ------------------------
    total = (
        lambda_data_sparse * data_sparse +
        lambda_multistep   * multistep_loss +
        lambda_takens      * takens_loss +
        lambda_param       * param_reg +
        lambda_energy      * energy_reg +
        lambda_chaos       * chaos_reg +
        lambda_u0          * u0_reg
    )

    return (total,
            data_sparse,
            multistep_loss,
            takens_loss,
            param_reg,
            energy_reg,
            chaos_reg,
            u0_reg,
            lam_est,
            U_pred,
            sigma, rho1, rho2, beta)

# ----------------------------
# Training loop
# ----------------------------
opt = optim.Adam(params, lr=5e-3)

losses = {k: [] for k in
          ["total", "data_sparse", "multistep", "takens",
           "param", "energy", "chaos", "u0"]}

best_loss = float("inf")
best_state = None
best_lam_est = None

for it in range(1, total_adam_steps + 1):
    opt.zero_grad()

    (total,
     Lds, Lms, Ltk,
     Lp, Le, Lc, Lu0,
     lam_est, U_pred,
     sigma_hat, rho1_hat, rho2_hat, beta_hat) = pinn_loss_abrupt_rho(
        log_sigma, log_rho1, log_rho2, log_beta,
        noise_std=process_noise_std
    )

    losses["total"].append(total.item())
    losses["data_sparse"].append(Lds.item())
    losses["multistep"].append(Lms.item())
    losses["takens"].append(Ltk.item())
    losses["param"].append(Lp.item())
    losses["energy"].append(Le.item())
    losses["chaos"].append(Lc.item())
    losses["u0"].append(Lu0.item())

    total.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()

    if total.item() < best_loss:
        best_loss = total.item()
        best_lam_est = lam_est.item()
        best_state = {
            "log_sigma": log_sigma.detach().clone().cpu(),
            "log_rho1":  log_rho1.detach().clone().cpu(),
            "log_rho2":  log_rho2.detach().clone().cpu(),
            "log_beta":  log_beta.detach().clone().cpu(),
            "u0":        u0_param.detach().clone().cpu()
        }

    if it == 1 or it % 100 == 0 or it == total_adam_steps:
        print(
            f"[{it:4d}/{total_adam_steps}] "
            f"tot={total.item():.3e} "
            f"sparse={Lds.item():.3e} "
            f"ms={Lms.item():.3e} "
            f"takens={Ltk.item():.3e} "
            f"param={Lp.item():.3e} "
            f"energy={Le.item():.3e} "
            f"chaos={Lc.item():.3e} "
            f"u0={Lu0.item():.3e} "
            f"| sigma={sigma_hat.item():.3f} "
            f"rho1={rho1_hat.item():.3f} "
            f"rho2={rho2_hat.item():.3f} "
            f"beta={beta_hat.item():.3f} "
            f"lam_max~{lam_est.item():.3f}"
        )

# Restore best
if best_state is not None:
    log_sigma.data = best_state["log_sigma"].to(device)
    log_rho1.data  = best_state["log_rho1"].to(device)
    log_rho2.data  = best_state["log_rho2"].to(device)
    log_beta.data  = best_state["log_beta"].to(device)
    u0_param.data  = best_state["u0"].to(device)

sigma_est = torch.exp(log_sigma).item()
rho1_est  = torch.exp(log_rho1).item()
rho2_est  = torch.exp(log_rho2).item()
beta_est  = torch.exp(log_beta).item()

print("\n===== Final estimated parameters (v2.3_abrupt_rho) =====")
print(f"sigma: true={sigma_true:.3f}, est={sigma_est:.3f}")
print(f"rho1:  true={rho1_true:.3f}, est={rho1_est:.3f}")
print(f"rho2:  true={rho2_true:.3f}, est={rho2_est:.3f}")
print(f"beta:  true={beta_true:.3f}, est={beta_est:.3f}")
print(f"u0_est (first traj) = {u0_param[0].detach().cpu().numpy()}")
if best_lam_est is not None:
    print(f"Approx. best λ_max during training ≈ {best_lam_est:.3f}")

# =============================================================================
#                                 PLOTS (PLOTLY)
# =============================================================================

# ===========================================================
#                Improved Loss Dashboard Plot
# ===========================================================

fig = make_subplots(rows=1, cols=1)

# Colours chosen for contrast & clarity
loss_colors = {
    "total":       "blue",
    "data_sparse": "orange",
    "multistep":   "green",
    "takens":      "purple",
    "param":       "red",
    "energy":      "brown",
    "chaos":       "cyan",
    "u0":          "black"
}

# Add each loss curve
for key in losses.keys():
    fig.add_trace(go.Scatter(
        y=np.array(losses[key]),
        mode="lines",
        name=key,
        line=dict(color=loss_colors[key], width=2)
    ))

# ---- Axis formatting ----
axis_style = dict(
    title_font=dict(size=20),
    tickfont=dict(size=14),
    showgrid=True,
    gridcolor="lightgrey",
    zeroline=False
)

fig.update_xaxes(
    title_text="Training Iteration",
    **axis_style
)

fig.update_yaxes(
    title_text="Loss (log scale)",
    type="log",
    **axis_style
)

# ---- Improve layout ----
fig.update_layout(
    title=dict(
        text="Inverse Lorenz-63 v2.3_abrupt_rho — Training Losses",
        font=dict(size=24)
    ),
    height=650,
    width=950,
    legend=dict(
        font=dict(size=14),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="lightgrey",
        borderwidth=1
    ),
    margin=dict(l=90, r=40, t=110, b=70),
    plot_bgcolor="white"
)

fig.show()

# ===========================================================
#          Additional Loss Plot (Linear Scale)
# ===========================================================

fig2 = make_subplots(rows=1, cols=1)

# Same colour scheme for consistency
loss_colors = {
    "total":       "blue",
    "data_sparse": "orange",
    "multistep":   "green",
    "takens":      "purple",
    "param":       "red",
    "energy":      "brown",
    "chaos":       "cyan",
    "u0":          "black"
}

# Add each loss curve
for key in losses.keys():
    fig2.add_trace(go.Scatter(
        y=np.array(losses[key]),
        mode="lines",
        name=key,
        line=dict(color=loss_colors[key], width=2)
    ))

# ---- Axis formatting ----
axis_style = dict(
    title_font=dict(size=20),
    tickfont=dict(size=14),
    showgrid=True,
    gridcolor="lightgrey",
    zeroline=False
)

fig2.update_xaxes(
    title_text="Training Iteration",
    **axis_style
)

fig2.update_yaxes(
    title_text="Loss (linear scale)",
    **axis_style
)

# ---- Layout styling ----
fig2.update_layout(
    title=dict(
        text="Inverse Lorenz-63 v2.3_abrupt_rho — Training Losses (Linear Scale)",
        font=dict(size=24)
    ),
    height=650,
    width=950,
    legend=dict(
        font=dict(size=14),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="lightgrey",
        borderwidth=1
    ),
    margin=dict(l=90, r=40, t=110, b=70),
    plot_bgcolor="white"
)

fig2.show()


# -------- Clean, polished Trajectory Comparison (k = 0) --------

# Choose which trajectory to plot
k_plot = 0

with torch.no_grad():
    U_hat = integrate_trajectories_abrupt_rho(
        u0_param,
        log_sigma, log_rho1, log_rho2, log_beta,
        t_raw, jump_idx,
        noise_std=0.0
    ).detach().cpu().numpy()  # [N,K,3]

# Extract true + predicted series
x_hat = U_hat[:, k_plot, 0]
y_hat = U_hat[:, k_plot, 1]
z_hat = U_hat[:, k_plot, 2]

x_true_k = X_true[k_plot, :, 0]
y_true_k = X_true[k_plot, :, 1]
z_true_k = X_true[k_plot, :, 2]
x_obs_k  = x_obs_full[k_plot]

# ---- Create subplot layout ----
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.10,
    
)

# ---- Axis style for consistent formatting ----
axis_style = dict(
    title_font=dict(size=18),
    tickfont=dict(size=14),
    showgrid=True,
    gridcolor="lightgrey",
    zeroline=False
)

# ================================
#               x(t)
# ================================
fig.add_trace(go.Scatter(
    x=t_full, y=x_true_k,
    name="True x", line=dict(color="blue")
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=t_full, y=x_hat,
    name="Pred x", line=dict(color="orange")
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=t_full[obs_idx_list[k_plot]],
    y=x_obs_k[obs_idx_list[k_plot]],
    mode="markers",
    name="Observed x",
    marker=dict(size=5, opacity=0.6, color="black")
), row=1, col=1)

fig.update_yaxes(title_text="x(t)", row=1, col=1, **axis_style)

# ================================
#               y(t)
# ================================
fig.add_trace(go.Scatter(
    x=t_full, y=y_true_k,
    name="True y", line=dict(color="blue")
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=t_full, y=y_hat,
    name="Pred y", line=dict(color="orange")
), row=2, col=1)

fig.update_yaxes(title_text="y(t)", row=2, col=1, **axis_style)

# ================================
#               z(t)
# ================================
fig.add_trace(go.Scatter(
    x=t_full, y=z_true_k,
    name="True z", line=dict(color="blue")
), row=3, col=1)

fig.add_trace(go.Scatter(
    x=t_full, y=z_hat,
    name="Pred z", line=dict(color="orange")
), row=3, col=1)

fig.update_yaxes(title_text="z(t)", row=3, col=1, **axis_style)

# ---- Shared X-axis label ----
fig.update_xaxes(title_text="Time (s)", row=3, col=1, **axis_style)

# ---- Vertical line marking end of training ----
fig.add_vline(
    x=t_train_end,
    line_width=2,
    line_dash="dash",
    line_color="black"
)

# ---- Layout adjustments ----
fig.update_layout(
    title=dict(
        text=f"Inverse Lorenz-63 v2.3_abrupt_rho — Time Series Trajectory",
        font=dict(size=22)
    ),
    height=950,
    width=950,
    showlegend=True,
    legend=dict(font=dict(size=14)),
    margin=dict(l=90, r=40, t=110, b=70),
    plot_bgcolor="white"
)

fig.show()


# -------- 3D attractor (all trajectories) --------
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=X_true[..., 0].ravel(),
    y=X_true[..., 1].ravel(),
    z=X_true[..., 2].ravel(),
    mode='markers',
    name='True (all traj)',
    marker=dict(size=1, opacity=0.4)
))
fig.add_trace(go.Scatter3d(
    x=U_hat[..., 0].ravel(),
    y=U_hat[..., 1].ravel(),
    z=U_hat[..., 2].ravel(),
    mode='markers',
    name='Inverse v2.3_abrupt_rho (all traj)',
    marker=dict(size=1, opacity=0.4)
))

fig.update_layout(
    title="Lorenz-63 attractor: True vs Inverse v2.3_abrupt_rho",
    width=900, height=700,
    scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="z"
    )
)
fig.show()
