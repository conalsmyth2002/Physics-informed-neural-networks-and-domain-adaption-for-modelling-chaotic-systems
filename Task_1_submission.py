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

Mackey–Glass + PINN + CT-MLP + Noise sweep
- Clean Mackey–Glass series
- PINN (SIREN) with physics prior
- CT-MLP (Takens ODE) dx/dt model
- Attractor stats and shadowing
- Gaussian noise sweep: sigma in [0, 0.3] (10 levels)
  * For each noise level, retrain:
      - PINN on noisy trajectory
      - CT-MLP on noisy dx/dt (Savitzky–Golay smoothing)
  * Evaluate vs clean truth
  * Plotly 3D slider over noise level (Clean vs PINN vs CT-MLP)
  * Plotly metric curves vs noise: RMSE, R², shadowing time

"""

import os, pathlib, webbrowser, time, json
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import welch, find_peaks, savgol_filter
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("CWD =", os.getcwd())

# ========= Repro & device =========
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

torch.set_default_dtype(torch.float32)

# ========= Mackey–Glass parameters =========
beta, gamma, n, tau = 0.2, 0.1, 10, 17
dt = 0.1
total_steps = 15000
x0 = 1.2

# ========= CT-MLP & noise sweep hyperparameters =========
CT_EPOCHS   = 1200
LR_MLP      = 1e-3
WD_MLP      = 1e-4
CT_PATIENCE = 80

# Noise sweep: σ multiple of std(x_clean) in [0, 0.3]
NOISE_MAX = 0.3
NOISE_LEVELS = np.linspace(0.0, NOISE_MAX, 10)


# Savitzky–Golay for noisy derivative
SAVGOL_WIN  = 21
SAVGOL_POLY = 3

# ========= Utilities =========

def generate_mg(beta: float, gamma: float, n: float, tau: float, dt: float,
                total_steps: int, x0: float) -> np.ndarray:
    """Generate Mackey–Glass with RK4 delay handling via stored history."""
    x = np.zeros(total_steps)
    lag = int(tau / dt)
    x[:lag] = x0
    def dxdt(x_t, x_tau):
        return beta * x_tau / (1 + x_tau**n) - gamma * x_t
    for t in range(lag, total_steps - 1):
        x_tau_val = x[t - lag]
        x_t = x[t]
        k1 = dxdt(x_t, x_tau_val)
        k2 = dxdt(x_t + 0.5 * dt * k1, x_tau_val)
        k3 = dxdt(x_t + 0.5 * dt * k2, x_tau_val)
        k4 = dxdt(x_t + dt * k3, x_tau_val)
        x[t + 1] = x_t + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x

# ========= Generate clean series & embedding indices =========
x = generate_mg(beta, gamma, n, tau, dt, total_steps, x0)
lag_steps   = int(tau / dt)
x_clean     = x[lag_steps:].copy()
T_end       = (len(x_clean) - 1) * dt

# Time grids
t_all_raw = np.arange(len(x_clean)) * dt   # 0..T_end
t_all_s   = t_all_raw / T_end              # 0..1

# Delay-embedding window (need 2τ history)
k0 = 2 * lag_steps
k1 = len(x_clean) - 1
K  = np.arange(k0, k1)

# ========= SIREN-based PINN =========
class SineLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 omega0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega0 = omega0
        self.is_first = is_first
        self.linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters()
    def reset_parameters(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1.0 / self.linear.in_features,
                    1.0 / self.linear.in_features
                )
            else:
                bound = np.sqrt(6 / self.linear.in_features) / self.omega0
                self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.zero_()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega0 * self.linear(x))

class SirenPINN(nn.Module):
    def __init__(self, hidden_dim: int = 64, hidden_layers: int = 3,
                 omega0_first: float = 30.0, omega0: float = 30.0):
        super().__init__()
        layers: List[nn.Module] = [SineLayer(1, hidden_dim, omega0_first, is_first=True)]
        for _ in range(hidden_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega0))
        self.final = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            b = np.sqrt(6 / hidden_dim) / omega0
            self.final.weight.uniform_(-b, b)
            self.final.bias.zero_()
        self.net = nn.Sequential(*layers, self.final)
    def forward(self, t_s: torch.Tensor) -> torch.Tensor:
        s = 2.0 * t_s - 1.0  # map [0,1] -> [-1,1]
        return self.net(s)

# ========= CT-MLP (Takens ODE) components =========

class MLP3(nn.Module):
    """Simple MLP 3->64->32->1 with Tanh; used as dx/dt approximator."""
    def __init__(self, d_in: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def train_regressor(
    model: nn.Module,
    Xtr_s: np.ndarray, ytr_s: np.ndarray,
    Xva_s: np.ndarray, yva_s: np.ndarray,
    epochs: int, lr: float, weight_decay: float, patience: int,
    loss_fn = nn.MSELoss()
) -> Tuple[nn.Module, List[float]]:
    """Generic training loop with simple early stopping on validation loss."""
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    Xtr = torch.tensor(Xtr_s, dtype=torch.float32, device=device)
    ytr = torch.tensor(ytr_s.reshape(-1, 1), dtype=torch.float32, device=device)
    loss_history: List[float] = []
    
    if Xva_s is not None and yva_s is not None:
        Xva = torch.tensor(Xva_s, dtype=torch.float32, device=device)
        yva = torch.tensor(yva_s.reshape(-1, 1), dtype=torch.float32, device=device)
    else:
        Xva = None
        yva = None
    
    best, wait, best_state = float('inf'), 0, None
    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        pred = model(Xtr); loss = loss_fn(pred, ytr)
        loss.backward(); opt.step()
        loss_history.append(float(loss.item()))

        if Xva is not None:
            model.eval()
            with torch.no_grad():
                v = loss_fn(model(Xva), yva).item()
        else:
            v = loss.item()

        if v < best - 1e-7:
            best, wait = v, 0
            best_state = {k: v_.detach().cpu().clone() for k, v_ in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, loss_history

def rollout_ct_mlp(deriv_model: nn.Module, xs: StandardScaler,
                   ys: StandardScaler, seed_series: np.ndarray) -> np.ndarray:
    """Integrate learned dx/dt = g([x, x_τ, x_2τ]) via RK4."""
    @torch.no_grad()
    def f_ct_dxdt_scaled(x_now, x_tau, x_2tau):
        feat = np.array([[x_now, x_tau, x_2tau]], dtype=np.float32)
        feat_s = xs.transform(feat)
        ydot_s = deriv_model(
            torch.tensor(feat_s, dtype=torch.float32, device=device)
        ).cpu().numpy().ravel()[0]
        return ys.inverse_transform([[ydot_s]]).ravel()[0]

    def rk4_update(x_val, f, h):
        k1 = f(x_val)
        k2 = f(x_val + 0.5*h*k1)
        k3 = f(x_val + 0.5*h*k2)
        k4 = f(x_val + h*k3)
        return x_val + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def step_one(state_arr, k_idx):
        x_t    = state_arr[k_idx]
        x_tau  = state_arr[k_idx - lag_steps]
        x_2tau = state_arr[k_idx - 2*lag_steps]
        def g(x_cur): return f_ct_dxdt_scaled(x_cur, x_tau, x_2tau)
        return rk4_update(x_t, g, dt)

    state = seed_series.copy()
    for k_idx in range(k0, len(seed_series) - 1):
        state[k_idx + 1] = step_one(state, k_idx)
    return state

def build_supervised_ct_features(x_series: np.ndarray) -> np.ndarray:
    """Build [x(t), x(t-τ), x(t-2τ)] for t indices in K."""
    X1 = x_series[k0:k1]
    X2 = x_series[k0 - lag_steps : k1 - lag_steps]
    X3 = x_series[k0 - 2*lag_steps : k1 - 2*lag_steps]
    return np.stack([X1, X2, X3], axis=1)

def _ensure_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1

def build_ct_targets_from_series(x_series: np.ndarray,
                                 use_savgol: bool = True) -> np.ndarray:
    """Estimate dx/dt at indices K, via Savitzky–Golay or central difference."""
    if use_savgol:
        L = len(x_series)
        max_odd = L if L % 2 == 1 else L - 1
        win = min(_ensure_odd(SAVGOL_WIN), max_odd)
        if win < 5: win = 5
        if win % 2 == 0: win += 1
        xdot_full = savgol_filter(
            x_series, window_length=win,
            polyorder=SAVGOL_POLY, deriv=1,
            delta=dt, mode='interp'
        )
        return xdot_full[K]
    else:
        return (x_series[K + 1] - x_series[K - 1]) / (2.0 * dt)

# ========= PINN training function =========

def train_pinn(t_obs_s: np.ndarray, y_obs: np.ndarray, *,

               epochs: int = 2000,
               collocation_points: int = 2000,
               ic_points: int = 400,           # kept for API compatibility, unused
               l_r: float = 1.0,
               l_ic: float = 10.0,             # unused (IC is hard)
               l_data: float = 0.8,
               lr: float = 5e-4,
               print_every: int = 200
               ) -> Tuple[SirenPINN, Tuple[float, float],
                          List[float], List[float], List[float], List[float]]:
    """
    PINN with hard history:
      * no IC loss term in the objective
      * PDE enforced only where t and (t-τ) are inside the NN domain (t ≥ 2τ)
    Returns:
      model, (mu_x, std_x),
      total_loss_history, Lr_history, Lic_history (all zeros), Ld_history
    """
    


    model = SirenPINN().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    # histories for plotting
    loss_hist: List[float] = []
    Lr_hist:   List[float] = []
    Lic_hist:  List[float] = []   # will be all zeros (hard IC)
    Ld_hist:   List[float] = []

    # ----- normalisation -----
    mu_x = float(np.mean(y_obs))
    std_x = float(np.std(y_obs) + 1e-8)
    mu_x_t  = torch.tensor(mu_x,  dtype=torch.float32, device=device)
    std_x_t = torch.tensor(std_x, dtype=torch.float32, device=device)

    t_train_s = torch.tensor(t_obs_s.reshape(-1, 1),
                             dtype=torch.float32, device=device)
    y_train   = torch.tensor(y_obs.reshape(-1, 1),
                             dtype=torch.float32, device=device)

    tau_s = tau / T_end

    def x_phys(t_s: torch.Tensor) -> torch.Tensor:
        return mu_x_t + std_x_t * model(t_s)

    def residual_loss() -> torch.Tensor:
        # enforce PDE only where both t and t-τ are in NN domain: t ≥ 2τ
        t_min = 2.0 * tau_s
        t_c = torch.rand(collocation_points, 1, device=device) * (1.0 - t_min) + t_min
        t_c.requires_grad_(True)

        x_t   = x_phys(t_c)
        x_tau = x_phys(t_c - tau_s)

        ones = torch.ones_like(x_t)
        dx_dt_s = torch.autograd.grad(
            x_t, t_c, grad_outputs=ones,
            create_graph=True, retain_graph=True
        )[0]
        dx_dt = dx_dt_s / T_end

        f = beta * x_tau / (1 + x_tau**n) - gamma * x_t
        return torch.mean((dx_dt - f) ** 2)

    def data_loss() -> torch.Tensor:
        x_d = x_phys(t_train_s)
        return nn.SmoothL1Loss()(x_d, y_train)

    best = float('inf'); best_state = None
    wait = 0; patience = 120

    for ep in range(1, epochs + 1):
        opt.zero_grad()

        k = min(1.0, ep / (0.5 * epochs))   # ramp PDE weight
        Lr  = residual_loss() * (l_r * (0.5 + 0.5 * k))
        Lic = torch.tensor(0.0, device=device)   # hard IC → no term
        Ld  = data_loss() * l_data

        loss = Lr + Ld
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # store histories
        loss_hist.append(float(loss.item()))
        Lr_hist.append(float(Lr.item()))
        Lic_hist.append(0.0)
        Ld_hist.append(float(Ld.item()))

        # early stopping on data MSE
        with torch.no_grad():
            v = (x_phys(t_train_s) - y_train).pow(2).mean().item()
        if v < best - 1e-7:
            best, wait = v, 0
            best_state = {k: v_.detach().cpu().clone()
                          for k, v_ in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

        if print_every and (ep % print_every == 0):
            print(f"[PINN] ep {ep}/{epochs}  "
                  f"Lr={Lr.item():.2e}  "
                  f"Ld={Ld.item():.2e}  Tot={loss.item():.2e}")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, (mu_x, std_x), loss_hist, Lr_hist, Lic_hist, Ld_hist


# ========= Train PINN on clean data =========
start = time.time()
obs_idx   = np.arange(k0 + 1, k1 + 1)  # region where delays are valid
obs_t_s   = t_all_s[obs_idx]
obs_y     = x_clean[obs_idx]

pinn, (mu_x, std_x), pinn_loss, pinn_Lr, pinn_Lic, pinn_Ld = train_pinn(
    obs_t_s, obs_y,
    epochs=2000, lr=5e-4, print_every=200
)

with torch.no_grad():
    t_s_full    = torch.tensor(t_all_s.reshape(-1, 1),
                               dtype=torch.float32, device=device)
    y_pinn_full = (mu_x + std_x * pinn(t_s_full).cpu().numpy().ravel())

print(f"PINN training done in {time.time() - start:.2f}s")

# ========= Train CT-MLP on clean data =========
start_ct = time.time()

X_ct_feat_clean = build_supervised_ct_features(x_clean)
xdot_clean      = build_ct_targets_from_series(x_clean, use_savgol=False)

idx_all_ct = np.arange(len(K))
idx_tr_ct, idx_va_ct = train_test_split(
    idx_all_ct, test_size=0.20, random_state=SEED, shuffle=True
)

xs_ct_clean = StandardScaler().fit(X_ct_feat_clean[idx_tr_ct])
ys_ct_clean = StandardScaler().fit(xdot_clean[idx_tr_ct].reshape(-1, 1))

X_tr_s_ct = xs_ct_clean.transform(X_ct_feat_clean[idx_tr_ct])
X_va_s_ct = xs_ct_clean.transform(X_ct_feat_clean[idx_va_ct])
y_tr_s_ct = ys_ct_clean.transform(xdot_clean[idx_tr_ct].reshape(-1, 1)).ravel()
y_va_s_ct = ys_ct_clean.transform(xdot_clean[idx_va_ct].reshape(-1, 1)).ravel()

mlp_ct_clean, ct_loss = train_regressor(
    MLP3(), X_tr_s_ct, y_tr_s_ct, X_va_s_ct, y_va_s_ct,
    epochs=CT_EPOCHS, lr=LR_MLP, weight_decay=WD_MLP,
    patience=CT_PATIENCE, loss_fn=nn.MSELoss()
)

state_ct_clean = rollout_ct_mlp(
    mlp_ct_clean, xs_ct_clean, ys_ct_clean, x_clean.copy()
)
y_ct_full_clean = state_ct_clean.copy()

ct_rmse_clean = float(np.sqrt(mean_squared_error(
    x_clean[obs_idx], y_ct_full_clean[obs_idx]
)))
ct_r2_clean   = float(r2_score(x_clean[obs_idx], y_ct_full_clean[obs_idx]))
print(f"CT-MLP (clean): RMSE={ct_rmse_clean:.4f}, R²={ct_r2_clean:.4f}")
print(f"CT-MLP clean training + rollout done in {time.time() - start_ct:.2f}s")

# ========= Embedding helpers =========

def embed3_arrays(series: np.ndarray, *, step: int = 2
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1 = series[k0:k1:step]
    x2 = series[k0 - lag_steps : k1 - lag_steps : step]
    x3 = series[k0 - 2 * lag_steps : k1 - 2 * lag_steps : step]
    return x1, x2, x3

def embed3_stack(series: np.ndarray, *, step: int = 2) -> np.ndarray:
    x1, x2, x3 = embed3_arrays(series, step=step)
    return np.stack([x1, x2, x3], axis=1)

# ========= Plotly 3D overlay (Clean vs PINN vs CT-MLP on clean) =========
_HAS_PLOTLY = False
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    if os.environ.get("DISPLAY") or os.name == "nt":
        pio.renderers.default = "browser"
    _HAS_PLOTLY = True
except Exception:
    print("Plotly not available or no display; will use static PNG only.")

x1_c,   x2_c,   x3_c   = embed3_arrays(x_clean)
x1_p,   x2_p,   x3_p   = embed3_arrays(y_pinn_full)
x1_ct0, x2_ct0, x3_ct0 = embed3_arrays(y_ct_full_clean)

def save_and_open_plotly(fig, filename: str):
    path = pathlib.Path.cwd() / filename
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"[Plotly] saved to: {path}")
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            webbrowser.open(f"file://{path}")
    except Exception as e:
        print(f"[Plotly] open failed: {e}")

if _HAS_PLOTLY:
    fig_clean = go.Figure()
    fig_clean.add_trace(go.Scatter3d(
        x=x1_c, y=x2_c, z=x3_c,
        mode="lines", name="True (clean)", line=dict(width=3)
    ))
    fig_clean.add_trace(go.Scatter3d(
        x=x1_p, y=x2_p, z=x3_p,
        mode="lines", name="PINN"
    ))
    fig_clean.add_trace(go.Scatter3d(
        x=x1_ct0, y=x2_ct0, z=x3_ct0,
        mode="lines", name="CT-MLP"
    ))
    fig_clean.update_layout(
        title="Mackey–Glass 3D: Clean vs PINN vs CT-MLP (clean training)",
        scene=dict(
            xaxis_title="x(t)", yaxis_title="x(t − τ)", zaxis_title="x(t − 2τ)",
            aspectmode="data"
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1.0),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    save_and_open_plotly(fig_clean, "mg_attractor_clean_vs_pinn_ctmlp.html")

# Static PNG
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
plt.figure(figsize=(8, 6))
ax = plt.axes(projection='3d')
ax.plot(x1_c,   x2_c,   x3_c,   label='True (clean)', linewidth=2)
ax.plot(x1_p,   x2_p,   x3_p,   label='PINN', alpha=0.9)
ax.plot(x1_ct0, x2_ct0, x3_ct0, label='CT-MLP', alpha=0.9)
ax.set_xlabel("x(t)"); ax.set_ylabel("x(t − τ)"); ax.set_zlabel("x(t − 2τ)")
ax.set_title("Mackey–Glass 3D: Clean vs PINN vs CT-MLP")
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig("mg_attractor_clean_vs_pinn_ctmlp.png", dpi=180)
plt.close()
print("Saved mg_attractor_clean_vs_pinn_ctmlp.png")

# ========= Attractor statistics =========

def mean_cov(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(X, axis=0)
    Sigma = np.cov(X, rowvar=False)
    return mu, Sigma

def acf_lags(x_arr: np.ndarray, lags: List[int]) -> Dict[int, float]:
    x0 = x_arr - x_arr.mean()
    acf = np.correlate(x0, x0, mode="full")
    acf = acf[acf.size // 2:]
    acf /= acf[0] + 1e-12
    out: Dict[int, float] = {}
    L = len(acf)
    for Lg in lags:
        if 0 <= Lg < L:
            out[int(Lg)] = float(acf[Lg])
    return out

def psd_peaks(x_arr: np.ndarray, fs: float, k: int = 3) -> List[Dict[str, float]]:
    x0 = x_arr - x_arr.mean()
    f, Pxx = welch(x0, fs=fs, nperseg=min(len(x0) // 2, 1024))
    peaks, _ = find_peaks(Pxx)
    if peaks.size == 0:
        return []
    idx = np.argsort(Pxx[peaks])[::-1][:k]
    return [{"freq": float(f[peaks][i]),
             "power": float(Pxx[peaks][i])} for i in idx]

def corr_dim_proxy(X: np.ndarray, *, max_points: int = 3000,
                   theiler_stride: Optional[int] = None,
                   r_quantiles: Tuple[float, float] = (0.05, 0.5)) -> float:

    if len(X) < 50:
        return float('nan')
    if theiler_stride is None:
        theiler_stride = max(1, lag_steps // 2)
    X_thin = X[::theiler_stride]
    if len(X_thin) > max_points:
        idx = np.linspace(0, len(X_thin) - 1,
                          max_points).astype(int)
        X_thin = X_thin[idx]
    d = pdist(X_thin, metric="euclidean")
    if d.size == 0:
        return float('nan')
    d_sorted = np.sort(d)
    r_min = np.quantile(d_sorted, r_quantiles[0])
    r_max = np.quantile(d_sorted, r_quantiles[1])
    if (not np.isfinite(r_min) or not np.isfinite(r_max) or
            r_min <= 0 or r_max <= r_min):
        return float('nan')
    rs = np.geomspace(r_min, r_max, 20)
    total_pairs = d.size
    Cr = np.array([(d <= r).sum() / total_pairs for r in rs]) + 1e-12
    lo, hi = int(0.2 * len(rs)), int(0.8 * len(rs))
    slope = np.polyfit(np.log(rs[lo:hi]), np.log(Cr[lo:hi]), 1)[0]
    return float(slope)

# Shadowing time (pred vs truth)

def shadowing_time(truth: np.ndarray, pred: np.ndarray, *,
                   start_idx: int = 0,
                   tol: Optional[float] = None,
                   frac_of_range: float = 0.05
                   ) -> Tuple[int, float]:
    """Return (shadow_steps, tol_used).
    If tol is None, use tol = frac_of_range * (range(truth[start:]))."""
    N = min(len(truth), len(pred))
    if start_idx >= N:
        return 0, float("nan")
    if tol is None:
        tr = truth[start_idx:N]
        tol = float(frac_of_range * (np.max(tr) - np.min(tr)))
    for i in range(start_idx, N):
        if abs(pred[i] - truth[i]) > tol:
            return i - start_idx, tol
    return N - start_idx, tol

def summarize_series(name: str, series: np.ndarray, *,
                     step: int = 2) -> Dict[str, object]:
    X = embed3_stack(series, step=step)
    mu, Sigma = mean_cov(X)
    acf = acf_lags(series[k0:k1:step],
                   lags=[1, 5, 10, 25, 50, 100,
                         lag_steps, 2 * lag_steps])
    peaks = psd_peaks(series[k0:k1:step], fs=1.0 / dt, k=3)
    fd = corr_dim_proxy(X)
    return {
        "name": name,
        "mean_x": float(mu[0]),
        "mean_x_tau": float(mu[1]),
        "mean_x_2tau": float(mu[2]),
        "cov_xx": float(Sigma[0, 0]),
        "cov_xx_tau": float(Sigma[0, 1]),
        "cov_xx_2tau": float(Sigma[0, 2]),
        "cov_tau_tau": float(Sigma[1, 1]),
        "cov_tau_2tau": float(Sigma[1, 2]),
        "cov_2tau_2tau": float(Sigma[2, 2]),
        "fractal_dim_proxy": fd,
        "acf_json": json.dumps(acf),
        "psd_peaks_json": json.dumps(peaks),
    }

# Base stats (clean)
stats_list = [
    summarize_series("True (clean)", x_clean),
    summarize_series("PINN (clean)", y_pinn_full),
    summarize_series("CT-MLP (clean)", y_ct_full_clean),
]
stats_df = pd.DataFrame(stats_list)


# Shadowing for PINN and CT-MLP vs clean
start_idx = 2 * lag_steps
sh_pinn_steps, sh_pinn_tol = shadowing_time(
    x_clean, y_pinn_full, start_idx=start_idx, tol=None, frac_of_range=0.10
)
sh_pinn_time = sh_pinn_steps * dt

sh_ct_steps, sh_ct_tol = shadowing_time(
    x_clean, y_ct_full_clean, start_idx=start_idx, tol=None, frac_of_range=0.10
)
sh_ct_time = sh_ct_steps * dt

for col in ("shadow_steps", "shadow_time", "shadow_tol"):
    if col not in stats_df.columns:
        stats_df[col] = np.nan

stats_df.loc[stats_df["name"] == "PINN (clean)",
             ["shadow_steps", "shadow_time", "shadow_tol"]] = [
    int(sh_pinn_steps), float(sh_pinn_time), float(sh_pinn_tol)
]
stats_df.loc[stats_df["name"] == "CT-MLP (clean)",
             ["shadow_steps", "shadow_time", "shadow_tol"]] = [
    int(sh_ct_steps), float(sh_ct_time), float(sh_ct_tol)
]

# Save stats
exports = pathlib.Path("exports")
exports.mkdir(parents=True, exist_ok=True)
stamp = time.strftime("%Y%m%d-%H%M%S")
stats_path = exports / f"attractor_stats_summary_{stamp}.csv"
stats_df.to_csv(stats_path, index=False)
print(f"Saved stats: {stats_path}")

try:
    stats_df.to_csv("attractor_stats_summary.csv", index=False)
    print("Updated: attractor_stats_summary.csv")
except Exception as e:
    print(f"[WARN] Could not write attractor_stats_summary.csv ({e})")

# ============================================================
#  FLATTEN ACF + PSD PEAKS INTO NEAT CSV FOR EXCEL
# ============================================================

def flatten_acf(acf_json_str: str) -> dict:
    """Convert {"lag": value, ...} into flat columns acf_<lag>=value."""
    acf = json.loads(acf_json_str)
    return {f"acf_{lag}": val for lag, val in acf.items()}

def flatten_psd(psd_json_str: str, max_peaks: int = 3) -> dict:
    """Convert list of {freq, power} peaks into columns."""
    peaks = json.loads(psd_json_str)
    out = {}
    for i in range(max_peaks):
        if i < len(peaks):
            out[f"psd{i+1}_freq"]  = peaks[i]["freq"]
            out[f"psd{i+1}_power"] = peaks[i]["power"]
        else:
            out[f"psd{i+1}_freq"]  = float("nan")
            out[f"psd{i+1}_power"] = float("nan")
    return out

# Build a new flattened table
rows = []

for _, row in stats_df.iterrows():
    base = row.to_dict()
    acf_flat = flatten_acf(row["acf_json"])
    psd_flat = flatten_psd(row["psd_peaks_json"], max_peaks=3)
    full_row = {**base, **acf_flat, **psd_flat}
    rows.append(full_row)

# New tidy DataFrame
stats_df_flat = pd.DataFrame(rows)

# Remove JSON columns
stats_df_flat = stats_df_flat.drop(columns=["acf_json", "psd_peaks_json"])

# Export
flat_path = exports / f"attractor_stats_flat_{stamp}.csv"
stats_df_flat.to_csv(flat_path, index=False)
print(f"Saved clean Excel-ready stats CSV to: {flat_path}")


# ========= Small 2D overlay (clean) =========
plt.figure(figsize=(9, 3))
L0, L1 = 500, min(12000, len(x_clean))
plt.plot(x_clean[L0:L1],       label='True (clean)')
plt.plot(y_pinn_full[L0:L1],   label='PINN (clean)', alpha=0.9)
plt.plot(y_ct_full_clean[L0:L1], label='CT-MLP (clean)', alpha=0.9)
plt.title('Time series overlay (clean subset)')
plt.xlabel('Time (t)'); plt.ylabel('x')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("time_overlay_clean_vs_pinn_ctmlp.png", dpi=160)
plt.show()
print("Saved time_overlay_clean_vs_pinn_ctmlp.png")

# =========================================================
#  3D STOCHASTIC BASIN ENTROPY (CT-MLP on Takens space)
# =========================================================

def build_seed_series_from_embedded_state(s0: float, s1: float, s2: float,
                                          horizon_steps: int) -> np.ndarray:
    """
    Build a synthetic history series for CT-MLP starting from a 3D Takens state
    (s0, s1, s2) ≈ (x(t), x(t-τ), x(t-2τ)).

    Time index mapping (discrete):
        idx = 0          -> t = -2τ      (x ≈ s2)
        idx = lag_steps  -> t = -τ       (x ≈ s1)
        idx = 2*lag_steps -> t = 0       (x ≈ s0)

    We linearly interpolate between these anchors to fill the history,
    then let rollout_ct_mlp evolve forward from t=0.
    """
    hist_len = 2 * lag_steps + 1
    total_len = hist_len + horizon_steps
    state = np.zeros(total_len, dtype=np.float32)

    i2 = 0
    i1 = lag_steps
    i0 = 2 * lag_steps

    # Anchor points
    state[i2] = s2
    state[i1] = s1
    state[i0] = s0

    # Linear interpolation between s2 -> s1 over [-2τ, -τ]
    if i1 > i2:
        for i in range(i2 + 1, i1):
            alpha = (i - i2) / float(i1 - i2)
            state[i] = (1 - alpha) * s2 + alpha * s1

    # Linear interpolation between s1 -> s0 over [-τ, 0]
    if i0 > i1:
        for i in range(i1 + 1, i0):
            alpha = (i - i1) / float(i0 - i1)
            state[i] = (1 - alpha) * s1 + alpha * s0

    # Past t=0 (indices > 2*lag_steps) will be filled by rollout_ct_mlp
    return state


def rollout_ct_mlp_from_emb_state(deriv_model: nn.Module,
                                  xs: StandardScaler,
                                  ys: StandardScaler,
                                  s0: float, s1: float, s2: float,
                                  horizon_steps: int) -> np.ndarray:
    """
    Rollout CT-MLP from a Takens-embedded initial state (s0,s1,s2).

    Returns trajectory x(t_k) for t >= 0, length horizon_steps+1.
    """
    seed_series = build_seed_series_from_embedded_state(
        s0, s1, s2, horizon_steps=horizon_steps
    )
    state_full = rollout_ct_mlp(deriv_model, xs, ys, seed_series)
    # indices k0 .. k0 + horizon_steps correspond to t in [0, horizon]
    return state_full[k0 : k0 + horizon_steps + 1]


def define_outcome_regions_from_data(series: np.ndarray,
                                     q1: float = 0.33,
                                     q2: float = 0.66) -> List[float]:
    """
    Define 3 outcome intervals [x_min,a), [a,b), [b,x_max] from empirical data.
    """
    s = series.copy()
    x_min = float(np.min(s))
    x_max = float(np.max(s))
    a = float(np.quantile(s, q1))
    b = float(np.quantile(s, q2))
    return [x_min, a, b, x_max]


def classify_outcome_series(x_traj: np.ndarray,
                            boundaries: List[float]) -> int:
    """
    Determine which region of x(t) is hit first.

    Regions:
        0: [x_min, a)
        1: [a, b)
        2: [b, x_max]

    Returns 0,1,2, or -1 if no region is visited.
    """
    x_min, a, b, x_max = boundaries
    for x_val in x_traj:
        if x_min <= x_val < a:
            return 0
        elif a <= x_val < b:
            return 1
        elif b <= x_val <= x_max:
            return 2
    return -1


def compute_stochastic_basin_entropy_3d_ct(
    s0_grid: np.ndarray,
    s1_grid: np.ndarray,
    s2_grid: np.ndarray,
    deriv_model: nn.Module,
    xs: StandardScaler,
    ys: StandardScaler,
    boundaries: List[float],
    *,
    K: int = 8,
    sigma: float = 0.05,
    horizon_time: float = 200.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    3D stochastic basin entropy in (s0,s1,s2) Takens space for CT-MLP.

    For each grid point center (s0,s1,s2):
      - sample K noisy points in R^3,
      - rollout CT-MLP for horizon_time,
      - classify first-hit region in x(t),
      - estimate probabilities p_k and normalized entropy S.
    """
    dt_local = dt  # from global
    horizon_steps = int(horizon_time / dt_local)
    N_out = 3

    Nx = len(s0_grid)
    Ny = len(s1_grid)
    Nz = len(s2_grid)

    S = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    probs = np.zeros((Nx, Ny, Nz, N_out), dtype=np.float32)

    total_cells = Nx * Ny * Nz
    cell = 0

    for i, s0 in enumerate(s0_grid):
        for j, s1 in enumerate(s1_grid):
            for k, s2 in enumerate(s2_grid):
                cell += 1
                counts = np.zeros(N_out, dtype=np.int32)

                # Gaussian perturbations in (s0,s1,s2)
                s0_samp = s0 + sigma * np.random.randn(K)
                s1_samp = s1 + sigma * np.random.randn(K)
                s2_samp = s2 + sigma * np.random.randn(K)

                for m in range(K):
                    x_traj = rollout_ct_mlp_from_emb_state(
                        deriv_model, xs, ys,
                        float(s0_samp[m]),
                        float(s1_samp[m]),
                        float(s2_samp[m]),
                        horizon_steps=horizon_steps,
                    )
                    outcome = classify_outcome_series(x_traj, boundaries)
                    if 0 <= outcome < N_out:
                        counts[outcome] += 1

                total = counts.sum()
                if total == 0:
                    S[i, j, k] = 0.0
                    probs[i, j, k, :] = 0.0
                else:
                    p = counts.astype(np.float64) / float(total)
                    probs[i, j, k, :] = p
                    nonzero = p[p > 0]
                    entropy = -np.sum(nonzero * np.log(nonzero))
                    S[i, j, k] = float(entropy / np.log(N_out))

                print(
                    f"[Basin3D CT] cell {cell}/{total_cells}  "
                    f"s0={s0:.2f}, s1={s1:.2f}, s2={s2:.2f}, "
                    f"counts={counts}, S={S[i,j,k]:.3f}"
                )

    return S, probs


def plot_basin3d_most_probable(
    s0_grid: np.ndarray,
    s1_grid: np.ndarray,
    s2_grid: np.ndarray,
    probs: np.ndarray,
    *,
    title: str = "CT-MLP: most probable outcome in Takens space",
):
    """3D scatter of most probable outcome region."""
    Nx, Ny, Nz, _ = probs.shape
    S0, S1, S2 = np.meshgrid(s0_grid, s1_grid, s2_grid, indexing="ij")
    most_prob = np.argmax(probs, axis=-1)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    x = S0.ravel()
    y = S1.ravel()
    z = S2.ravel()
    c = most_prob.ravel()

    sc = ax.scatter(x, y, z, c=c, s=30)
    ax.set_xlabel("x(t) = s0")
    ax.set_ylabel("x(t-τ) = s1")
    ax.set_zlabel("x(t-2τ) = s2")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label="Outcome index")
    plt.tight_layout()
    plt.savefig("basin3d_ctmlp_most_probable.png", dpi=180)
    print("Saved basin3d_ctmlp_most_probable.png")


def plot_basin3d_entropy_slice(
    s0_grid: np.ndarray,
    s1_grid: np.ndarray,
    s2_grid: np.ndarray,
    S: np.ndarray,
    *,
    fixed_axis: str = "s2",
    fixed_index: Optional[int] = None,
    title_prefix: str = "CT-MLP basin entropy slice",
):
    """
    Plot a 2D slice of S for visualization.
    fixed_axis ∈ {"s0","s1","s2"}; if fixed_index is None, middle index used.
    """
    Nx, Ny, Nz = S.shape

    if fixed_axis == "s2":
        if fixed_index is None:
            fixed_index = Nz // 2
        S_sl = S[:, :, fixed_index]
        xx, yy = np.meshgrid(s0_grid, s1_grid, indexing="ij")
        label = f"s2 = {s2_grid[fixed_index]:.3f}"
        xlab, ylab = "s0", "s1"
    elif fixed_axis == "s1":
        if fixed_index is None:
            fixed_index = Ny // 2
        S_sl = S[:, fixed_index, :]
        xx, yy = np.meshgrid(s0_grid, s2_grid, indexing="ij")
        label = f"s1 = {s1_grid[fixed_index]:.3f}"
        xlab, ylab = "s0", "s2"
    elif fixed_axis == "s0":
        if fixed_index is None:
            fixed_index = Nx // 2
        S_sl = S[fixed_index, :, :]
        xx, yy = np.meshgrid(s1_grid, s2_grid, indexing="ij")
        label = f"s0 = {s0_grid[fixed_index]:.3f}"
        xlab, ylab = "s1", "s2"
    else:
        raise ValueError("fixed_axis must be one of 's0','s1','s2'")

    plt.figure(figsize=(6, 5))
    cp = plt.contourf(xx, yy, S_sl, levels=20)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(f"{title_prefix} ({label})")
    plt.colorbar(cp, label="Normalized basin entropy")
    plt.tight_layout()
    plt.savefig(
        f"basin3d_ctmlp_entropy_slice_{fixed_axis}.png",
        dpi=180,
    )
    print(f"Saved basin3d_ctmlp_entropy_slice_{fixed_axis}.png")

# ========= Export attractor point clouds (clean) =========

def build_embedding(series: np.ndarray, label: str, ds: int = 1) -> pd.DataFrame:
    i = np.arange(k0, k1, ds)
    return pd.DataFrame({
        "t_idx": i,
        "t": (i - k0) * dt,
        "x_t": series[i],
        "x_tau": series[i - lag_steps],
        "x_2tau": series[i - 2 * lag_steps],
        "source": label,
    })

df_clean = build_embedding(x_clean,         "clean",   ds=1)
df_pinn  = build_embedding(y_pinn_full,    "pinn",    ds=1)
df_ct    = build_embedding(y_ct_full_clean,"ct_mlp",  ds=1)

df_points_clean = pd.concat([df_clean, df_pinn, df_ct], ignore_index=True)
base_clean = exports / f"mg_attractor_points_clean_{stamp}"

csv_path = str(base_clean) + ".csv"
df_points_clean.to_csv(csv_path, index=False)
print(f"[EXPORT] Clean CSV -> {csv_path}")

try:
    pq_path = str(base_clean) + ".parquet"
    df_points_clean.to_parquet(pq_path, index=False)
    print(f"[EXPORT] Clean Parquet -> {pq_path}")
except Exception as e:
    print(f"[EXPORT] Clean Parquet skipped ({e})")

npz_path = str(base_clean) + ".npz"
np.savez(
    npz_path,
    t_idx=df_points_clean["t_idx"].to_numpy(),
    t=df_points_clean["t"].to_numpy(dtype=np.float64),
    x_t=df_points_clean["x_t"].to_numpy(dtype=np.float64),
    x_tau=df_points_clean["x_tau"].to_numpy(dtype=np.float64),
    x_2tau=df_points_clean["x_2tau"].to_numpy(dtype=np.float64),
    source=df_points_clean["source"].astype("U").to_numpy(),
)
print(f"[EXPORT] Clean NPZ -> {npz_path}")

# ========= Console summary (clean) =========
print("========== CLEAN ATTRACTOR STATISTICS ==========")
with pd.option_context('display.max_columns', None,
                       'display.width', 160,
                       'display.max_colwidth', 120):
    print(stats_df)
print("==========================================")

for _, row in stats_df.iterrows():
    print(f"--- {row['name']} ---")
    print(f"Mean x: {row['mean_x']:.4f}, Cov_xx: {row['cov_xx']:.4e}, "
          f"Fractal dim proxy: {row['fractal_dim_proxy']:.3f}")
    print("ACF:", json.loads(row['acf_json']))
    print("Top PSD peaks:", json.loads(row['psd_peaks_json']))
    if row['name'] in ('PINN (clean)', 'CT-MLP (clean)'):
        print(f"Shadowing: steps={int(row['shadow_steps']) if pd.notna(row['shadow_steps']) else 'NA'}, "
              f"time={float(row['shadow_time']) if pd.notna(row['shadow_time']) else np.nan:.3f}, "
              f"tol={float(row['shadow_tol']) if pd.notna(row['shadow_tol']) else np.nan:.3g}")



# =========================================================
#  NOISE SWEEP: retrain PINN & CT-MLP on noisy data
# =========================================================

print("\n=== Running Noise Sweep (sigma in [0, 0.3], 10 levels) ===")
sigma_clean = np.std(x_clean)
rng = np.random.default_rng(123)

noise_results = []
pinn_noise_trajs = {}
ct_noise_trajs   = {}

for sigma_mult in NOISE_LEVELS:
    sigma = sigma_mult * sigma_clean
    print(f"\n--- Noise level {sigma_mult:.3f} (σ={sigma:.4f}) ---")

    # Noisy trajectory
    x_noisy = x_clean + rng.normal(0.0, sigma, size=x_clean.shape)

    # ----- 1) PINN retrained on noisy data -----
    obs_y_noisy = x_noisy[obs_idx]
    pinn_n, (mu_x_n, std_x_n), pinn_loss_n, pinn_Lr_n, pinn_Lic_n, pinn_Ld_n = train_pinn(
        obs_t_s, obs_y_noisy,
        epochs=1200, collocation_points=2000, ic_points=300,
        l_r=1.0, l_ic=10.0, l_data=0.7, lr=5e-4, print_every=0
    )

    with torch.no_grad():
        y_pinn_full_n = (mu_x_n +
            std_x_n * pinn_n(t_s_full).cpu().numpy().ravel())

    pinn_noise_trajs[float(sigma_mult)] = y_pinn_full_n

    y_true = x_clean[obs_idx]
    y_pred_pinn_n = y_pinn_full_n[obs_idx]
    rmse_pinn_n = float(np.sqrt(
        mean_squared_error(y_true, y_pred_pinn_n)
    ))
    r2_pinn_n   = float(r2_score(y_true, y_pred_pinn_n))
    sh_steps_pinn_n, sh_tol_pinn_n = shadowing_time(
        x_clean, y_pinn_full_n, start_idx=start_idx,
        tol=None, frac_of_range=0.1
    )
    sh_time_pinn_n = sh_steps_pinn_n * dt

    # ----- 2) CT-MLP retrained on noisy dx/dt via Savitzky–Golay -----
    X_ct_feat_noisy = build_supervised_ct_features(x_noisy)
    xdot_noisy      = build_ct_targets_from_series(
        x_noisy, use_savgol=True
    )

    xs_ct_n = StandardScaler().fit(X_ct_feat_noisy[idx_tr_ct])
    ys_ct_n = StandardScaler().fit(xdot_noisy[idx_tr_ct].reshape(-1, 1))

    X_tr_s_n = xs_ct_n.transform(X_ct_feat_noisy[idx_tr_ct])
    X_va_s_n = xs_ct_n.transform(X_ct_feat_noisy[idx_va_ct])
    y_tr_s_n = ys_ct_n.transform(xdot_noisy[idx_tr_ct].reshape(-1, 1)).ravel()
    y_va_s_n = ys_ct_n.transform(xdot_noisy[idx_va_ct].reshape(-1, 1)).ravel()

    mlp_ct_n, _ = train_regressor(
        MLP3(), X_tr_s_n, y_tr_s_n, X_va_s_n, y_va_s_n,
        epochs=800, lr=LR_MLP, weight_decay=WD_MLP,
        patience=80, loss_fn=nn.SmoothL1Loss()
    )

    state_ct_n = rollout_ct_mlp(
        mlp_ct_n, xs_ct_n, ys_ct_n, x_noisy.copy()
    )
    ct_noise_trajs[float(sigma_mult)] = state_ct_n

    y_pred_ct_n = state_ct_n[obs_idx]
    rmse_ct_n = float(np.sqrt(
        mean_squared_error(y_true, y_pred_ct_n)
    ))
    r2_ct_n   = float(r2_score(y_true, y_pred_ct_n))
    sh_steps_ct_n, sh_tol_ct_n = shadowing_time(
        x_clean, state_ct_n, start_idx=start_idx,
        tol=None, frac_of_range=0.1
    )
    sh_time_ct_n = sh_steps_ct_n * dt

    noise_results.append({
        "noise_sigma_mult": float(sigma_mult),
        "noise_sigma_abs":  float(sigma),
        "rmse_pinn": rmse_pinn_n,
        "r2_pinn":   r2_pinn_n,
        "shadow_steps_pinn": sh_steps_pinn_n,
        "shadow_time_pinn":  sh_time_pinn_n,
        "rmse_ct": rmse_ct_n,
        "r2_ct":   r2_ct_n,
        "shadow_steps_ct": sh_steps_ct_n,
        "shadow_time_ct":  sh_time_ct_n,
    })

# Save noise sweep results
df_noise = pd.DataFrame(noise_results)
noise_path = exports / f"noise_sweep_results_{stamp}.csv"
df_noise.to_csv(noise_path, index=False)
print(f"\nSaved noise sweep results: {noise_path}")
print("\n=== Noise sweep summary ===")
print(df_noise)

# ========= Plotly noise slider: 3D attractor vs noise =========
if _HAS_PLOTLY:
    frames = []
    first_sigma = float(NOISE_LEVELS[0])
    x1_p_first, x2_p_first, x3_p_first = embed3_arrays(
        pinn_noise_trajs[first_sigma]
    )
    x1_ct_first, x2_ct_first, x3_ct_first = embed3_arrays(
        ct_noise_trajs[first_sigma]
    )

    for sigma_mult in NOISE_LEVELS:
        sig = float(sigma_mult)
        y_pinn_sigma = pinn_noise_trajs[sig]
        y_ct_sigma   = ct_noise_trajs[sig]

        x1_p_s, x2_p_s, x3_p_s = embed3_arrays(y_pinn_sigma)
        x1_ct_s, x2_ct_s, x3_ct_s = embed3_arrays(y_ct_sigma)

        frames.append(go.Frame(
            name=f"sigma={sig:.3f}",
            data=[
                go.Scatter3d(
                    x=x1_c, y=x2_c, z=x3_c,
                    mode="lines", name="True (clean)", line=dict(width=3)
                ),
                go.Scatter3d(
                    x=x1_p_s, y=x2_p_s, z=x3_p_s,
                    mode="lines", name="PINN (noisy)"
                ),
                go.Scatter3d(
                    x=x1_ct_s, y=x2_ct_s, z=x3_ct_s,
                    mode="lines", name="CT-MLP (noisy)"
                )
            ],
            traces=[0, 1, 2]
        ))

    fig_noise = go.Figure(
        data=[
            go.Scatter3d(
                x=x1_c, y=x2_c, z=x3_c,
                mode="lines", name="True (clean)", line=dict(width=3)
            ),
            go.Scatter3d(
                x=x1_p_first, y=x2_p_first, z=x3_p_first,
                mode="lines", name="PINN (noisy)"
            ),
            go.Scatter3d(
                x=x1_ct_first, y=x2_ct_first, z=x3_ct_first,
                mode="lines", name="CT-MLP (noisy)"
            ),
        ],
        frames=frames
    )

    steps = []
    for f in fig_noise.frames:
        steps.append(dict(
            method="animate",
            args=[[f.name], {
                "frame": {"duration": 0, "redraw": True},
                "mode": "immediate",
                "transition": {"duration": 0}
            }],
            label=f.name.replace("sigma=", "σ=")
        ))

    fig_noise.update_layout(
        title="Mackey–Glass 3D: Clean vs PINN vs CT-MLP across noise σ",
        scene=dict(
            xaxis_title="x(t)",
            yaxis_title="x(t − τ)",
            zaxis_title="x(t − 2τ)",
            aspectmode="data"
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1.0),
        margin=dict(l=0, r=0, b=0, t=40),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.08, x=0,
            xanchor="left", yanchor="top",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, {
                         "fromcurrent": True,
                         "frame": {"duration": 400, "redraw": True},
                         "transition": {"duration": 200}
                     }]),
                dict(label="Pause", method="animate",
                     args=[[None], {
                         "frame": {"duration": 0, "redraw": False},
                         "mode": "immediate",
                         "transition": {"duration": 0}
                     }]),
            ]
        )],
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Noise ", "visible": True},
            pad={"t": 10, "b": 0},
            steps=steps
        )]
    )

    save_and_open_plotly(fig_noise,
                         "mg_attractor_noise_slider_pinn_ctmlp.html")

    # ========= Plotly metric curves vs noise =========
    fig_rmse = go.Figure()
    fig_rmse.add_trace(go.Scatter(
        x=df_noise["noise_sigma_mult"], y=df_noise["rmse_pinn"],
        mode="lines+markers", name="PINN RMSE"
    ))
    fig_rmse.add_trace(go.Scatter(
        x=df_noise["noise_sigma_mult"], y=df_noise["rmse_ct"],
        mode="lines+markers", name="CT-MLP RMSE"
    ))
    fig_rmse.update_layout(
        title="RMSE vs noise σ (multiple of std)",
        xaxis_title="σ / std(x_clean)",
        yaxis_title="RMSE",
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1.0)
    )
    save_and_open_plotly(fig_rmse, "noise_rmse_pinn_ctmlp.html")

    fig_r2 = go.Figure()
    fig_r2.add_trace(go.Scatter(
        x=df_noise["noise_sigma_mult"], y=df_noise["r2_pinn"],
        mode="lines+markers", name="PINN R²"
    ))
    fig_r2.add_trace(go.Scatter(
        x=df_noise["noise_sigma_mult"], y=df_noise["r2_ct"],
        mode="lines+markers", name="CT-MLP R²"
    ))
    fig_r2.update_layout(
        title="R² vs noise σ (multiple of std)",
        xaxis_title="σ / std(x_clean)",
        yaxis_title="R²",
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1.0)
    )
    save_and_open_plotly(fig_r2, "noise_r2_pinn_ctmlp.html")

    fig_shadow = go.Figure()
    fig_shadow.add_trace(go.Scatter(
        x=df_noise["noise_sigma_mult"],
        y=df_noise["shadow_time_pinn"],
        mode="lines+markers", name="PINN shadow time"
    ))
    fig_shadow.add_trace(go.Scatter(
        x=df_noise["noise_sigma_mult"],
        y=df_noise["shadow_time_ct"],
        mode="lines+markers", name="CT-MLP shadow time"
    ))
    fig_shadow.update_layout(
        title="Shadowing time vs noise σ (multiple of std)",
        xaxis_title="σ / std(x_clean)",
        yaxis_title="Shadowing time",
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1.0)
    )
    save_and_open_plotly(fig_shadow,
                         "noise_shadow_pinn_ctmlp.html")

# Total training losses (clean)
plt.figure(figsize=(6, 4))
plt.plot(pinn_loss, label="PINN total loss")
plt.plot(ct_loss, label="CT-MLP loss")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curves for PINN and CT-MLP")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pinn_ctmlp_training_loss.png", dpi=150)
plt.show()

# PINN loss decomposition: which term spikes?
plt.figure(figsize=(6, 4))
plt.plot(pinn_Lr,  label="PDE residual Lr")
plt.plot(pinn_Lic, label="IC loss Lic")
plt.plot(pinn_Ld,  label="Data loss Ld")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss component")
plt.title("PINN Loss Decomposition")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pinn_loss_decomposition.png", dpi=150)
plt.show()


# =========================================================
#  RUN 3D BASIN ENTROPY (toggle with RUN_BASIN3D)
# =========================================================
RUN_BASIN3D = False  # <-- set to True when you want to run it

if RUN_BASIN3D:
    np.random.seed(0)

    # Use the clean CT-MLP model & scalers
    deriv_model_ct = mlp_ct_clean
    xs_ct = xs_ct_clean
    ys_ct = ys_ct_clean

    # Outcome regions derived from clean truth (x_clean)
    boundaries_ct = define_outcome_regions_from_data(
        x_clean[k0:k1]  # same window used for embedding
    )
    print("Basin outcome boundaries (x):", boundaries_ct)

    # 3D grid in Takens space; ranges from empirical x_clean
    x_min = float(np.min(x_clean[k0:k1]))
    x_max = float(np.max(x_clean[k0:k1]))

    # You can tweak N_axis for resolution vs runtime
    N_axis = 8  # 8^3 = 512 cells (already nontrivial)
    s0_grid = np.linspace(x_min, x_max, N_axis)
    s1_grid = np.linspace(x_min, x_max, N_axis)
    s2_grid = np.linspace(x_min, x_max, N_axis)

    # Basin entropy parameters
    K_samples = 6        # noisy samples per cell
    sigma_init = 0.05    # noise in Takens space
    horizon_T = 200.0    # physical time horizon

    S_basin, probs_basin = compute_stochastic_basin_entropy_3d_ct(
        s0_grid, s1_grid, s2_grid,
        deriv_model_ct, xs_ct, ys_ct,
        boundaries_ct,
        K=K_samples,
        sigma=sigma_init,
        horizon_time=horizon_T,
    )

    # Save raw arrays
    np.savez(
        "basin3d_ctmlp_results.npz",
        s0_grid=s0_grid,
        s1_grid=s1_grid,
        s2_grid=s2_grid,
        S=S_basin,
        probs=probs_basin,
        boundaries=np.array(boundaries_ct, dtype=np.float32),
    )
    print("Saved basin3d_ctmlp_results.npz")

    # Plots: 3D most probable outcome + one entropy slice
    plot_basin3d_most_probable(s0_grid, s1_grid, s2_grid, probs_basin)
    plot_basin3d_entropy_slice(
        s0_grid, s1_grid, s2_grid, S_basin,
        fixed_axis="s2", fixed_index=None,
    )


print("\nAll done.")
