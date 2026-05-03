"""
Gradient-flow simulation for E=2 linear-router MoE with Gaussian mixture data.

Data model:  x^μ = κ·U_{c_μ} + ξ^μ,   ξ ~ N(0, I_d)
Teacher:     y^μ = φ(U_{c_μ}·x^μ)   =  φ(κ + noise)

Hardware acceleration
─────────────────────
Automatically selects the fastest available backend:
  • CUDA  (NVIDIA GPU)
  • MPS   (Apple Silicon GPU via Metal)
  • CPU   (NumPy-based, identical numerics)

Pass  device='cuda' | 'mps' | 'cpu'  to override, or device=None for auto.

Memory scaling for X ∈ ℝ^{n×d} at float32:
  d=6,000  →  0.14 GB   (fits anywhere)
  d=20,000 →  1.6 GB    (consumer GPU, e.g. RTX 3090 / M1 Pro)
  d=60,000 →  14.4 GB   (A100-80GB / M2-Ultra)

chunk_size controls peak activation memory during the X @ W.T matmul;
reduce it if you hit OOM (default 4096 rows at a time).
"""

from __future__ import annotations

import math
import torch
import numpy as np
from scipy.special import erf
from tqdm import trange

# ─── numpy reference phi (kept for callers that import it from here) ──────────
phi_np = lambda z: erf(z / math.sqrt(2.0))
phi    = phi_np          # backward-compat alias


# ─── Device selection ─────────────────────────────────────────────────────────

def _resolve_device(device) -> torch.device:
    """Return the best available device, or honour an explicit string request."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── Differentiable primitives ────────────────────────────────────────────────

def _phi(z: torch.Tensor) -> torch.Tensor:
    """φ(z) = erf(z/√2)."""
    return torch.erf(z * (1.0 / math.sqrt(2.0)))


def _phi_prime(z: torch.Tensor) -> torch.Tensor:
    """φ'(z) = √(2/π)·exp(-z²/2)."""
    return math.sqrt(2.0 / math.pi) * torch.exp(-0.5 * z * z)


# ─── Memory-safe matmul ───────────────────────────────────────────────────────

def _mm_chunk(A: torch.Tensor, B: torch.Tensor, chunk: int) -> torch.Tensor:
    """Compute A @ B in chunks along dim-0 of A to cap peak memory.

    A: (n, d), B: (d, k) → (n, k)
    Set chunk=0 to disable chunking.
    """
    if chunk <= 0 or chunk >= A.shape[0]:
        return A @ B
    parts = [A[s:s + chunk] @ B for s in range(0, A.shape[0], chunk)]
    return torch.cat(parts, dim=0)


# ─── Main simulation ──────────────────────────────────────────────────────────

def run_gf_linear_mu(
    seed: int        = 0,
    alpha_bar: float = 1.0,
    m: int           = 300,
    d: int           = 1200,
    T_max: float     = 8.0,
    eta_gf: float    = 0.1,
    a0: float        = 0.5,
    P_self0: float   = 0.2,
    Rv_self0: float  = 0.05,
    a1: float        = 0.3,
    kappa: float     = 1.0,
    device           = None,
    chunk_size: int  = 4096,
    dtype: torch.dtype = torch.float32,
):
    """
    Gradient-flow simulation for the E=2 linear-router MoE.

    All heavy matrix algebra is executed on `device` (CUDA / MPS / CPU).
    Only scalar statistics are transferred back to CPU each step.

    Parameters
    ----------
    seed       : RNG seed — results are identical across devices.
    alpha_bar  : sample ratio n = round(α·d).
    m          : neurons per expert.
    d          : feature dimension.
    T_max      : total gradient-flow time.
    eta_gf     : step size.
    a0         : initial second-layer weight.
    P_self0    : initial expert-teacher overlap W[e]·U[e].
    Rv_self0   : initial router-teacher overlap V[e]·U[e].
    a1         : linear router coefficient  g_e = a0 ± a1·disc.
    kappa      : cluster-mean magnitude  μ_c = κ·U_c.
    device     : 'cuda' | 'mps' | 'cpu' | None (auto).
    chunk_size : rows of X per matmul call; reduce if you hit OOM.
    dtype      : torch.float32 (default) or torch.float64.

    Returns
    -------
    dict with keys: 't', 'a', 'P_self', 'P_cross', 'Rv_self', 'Rv_cross',
                    'Delta', 'Delta_v', 'Pv_self', 'Pv_cross', 'train_loss'
    """
    dev     = _resolve_device(device)
    E       = 2
    n       = int(round(alpha_bar * d))
    norm    = float(d)          # 1/d normalisation (DMFT convention)
    gp      = a1 / math.sqrt(2.0)
    N_steps = int(T_max / eta_gf) + 1

    # ------------------------------------------------------------------
    # Data generation — always done on CPU (numpy) for cross-device
    # reproducibility, then moved to `dev` in one transfer.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)

    # Teacher directions: U[e] = e_e  (standard basis, one-hot)
    U_np = np.zeros((E, d), dtype=np.float64)
    for e in range(E):
        U_np[e, e] = 1.0

    # Data:  x^μ = κ·U_{c_μ} + ξ^μ
    cluster_np = np.arange(n) % E
    X_np = rng.standard_normal((n, d))
    for c in range(E):
        X_np[cluster_np == c] += kappa * U_np[c]

    X       = torch.tensor(X_np, dtype=dtype, device=dev)          # (n, d)
    cluster = torch.tensor(cluster_np, dtype=torch.long, device=dev)
    del X_np     # free CPU copy once on device

    # Teacher labels  y^μ = φ(U_{c_μ}·x^μ) = φ(X[i, cluster[i]])
    # U is one-hot so U_{c}·x = x[c] (a single column).
    y = _phi(X[torch.arange(n, device=dev), cluster])               # (n,)

    # Move U to device (only needed for init)
    U = torch.tensor(U_np, dtype=dtype, device=dev)                 # (E, d)

    # ------------------------------------------------------------------
    # Weight initialisation
    # Expert weights:  project out all U_e directions first so P_cross=0.
    # Router weights:  Gram-Schmidt orthogonalisation across experts.
    # ------------------------------------------------------------------
    def _init_expert(e: int) -> np.ndarray:
        rand = rng.standard_normal((m, d))
        for ep in range(E):
            rand -= (rand @ U_np[ep])[:, None] * U_np[ep]
        rand /= np.linalg.norm(rand, axis=1, keepdims=True)
        w = P_self0 * U_np[e] + math.sqrt(max(0.0, 1.0 - P_self0 ** 2)) * rand
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        return w.astype(np.float64)

    def _init_router(e: int, prev: list[np.ndarray]) -> np.ndarray:
        v = rng.standard_normal(d)
        for ep in range(E):
            v -= (v @ U_np[ep]) * U_np[ep]
        for vp in prev:
            v -= (v @ vp) * vp
        v /= np.linalg.norm(v)
        v = Rv_self0 * U_np[e] + math.sqrt(max(0.0, 1.0 - Rv_self0 ** 2)) * v
        v /= np.linalg.norm(v)
        return v.astype(np.float64)

    # W: stacked expert weights (E*m, d)
    W = torch.tensor(
        np.vstack([_init_expert(e) for e in range(E)]),
        dtype=dtype, device=dev,
    )

    # V: router weights (E, d)
    prev_v: list[np.ndarray] = []
    V_rows = []
    for e in range(E):
        vv = _init_router(e, prev_v)
        prev_v.append(vv)
        V_rows.append(vv)
    V = torch.tensor(np.stack(V_rows), dtype=dtype, device=dev)    # (E, d)

    a_vec = torch.full((E,), a0, dtype=dtype, device=dev)

    # Free device memory we no longer need
    del U

    # ------------------------------------------------------------------
    # Output storage (pre-allocated on CPU)
    # ------------------------------------------------------------------
    t_arr          = np.arange(N_steps, dtype=np.float64) * eta_gf
    a_arr          = np.empty(N_steps, dtype=np.float64)
    P_self_arr     = np.empty(N_steps, dtype=np.float64)
    P_cross_arr    = np.empty(N_steps, dtype=np.float64)
    Rv_self_arr    = np.empty(N_steps, dtype=np.float64)
    Rv_cross_arr   = np.empty(N_steps, dtype=np.float64)
    Pv_self_arr    = np.empty(N_steps, dtype=np.float64)
    Pv_cross_arr   = np.empty(N_steps, dtype=np.float64)
    train_loss_arr = np.empty(N_steps, dtype=np.float64)

    def _record(s: int, loss_val: float | None = None):
        """Extract all scalar order-parameters — minimal CPU↔device traffic."""
        # P_self: mean of W[e]·U[e] = W[e,:,e] (one-hot U, so just a column).
        # W[:m, 0] = W[0, :, U_0-dim=0].  W[m:, 1] for expert 1.
        ps  = 0.5 * (float(W[:m, 0].mean()) + float(W[m:, 1].mean()))
        pc  = 0.5 * (float(W[:m, 1].mean()) + float(W[m:, 0].mean()))
        rvs = 0.5 * (float(V[0, 0]) + float(V[1, 1]))
        rvc = 0.5 * (float(V[0, 1]) + float(V[1, 0]))
        # Pv: (m, d) @ (d,) → (m,); mean over m
        pvs = 0.5 * (float((W[:m] @ V[0]).mean()) + float((W[m:] @ V[1]).mean()))
        pvc = 0.5 * (float((W[:m] @ V[1]).mean()) + float((W[m:] @ V[0]).mean()))

        a_arr[s]          = float(a_vec.mean())
        P_self_arr[s]     = ps
        P_cross_arr[s]    = pc
        Rv_self_arr[s]    = rvs
        Rv_cross_arr[s]   = rvc
        Pv_self_arr[s]    = pvs
        Pv_cross_arr[s]   = pvc
        if loss_val is not None:
            train_loss_arr[s] = loss_val

    _record(0)

    # ------------------------------------------------------------------
    # Pre-compute step-invariant scalars
    # ------------------------------------------------------------------
    inv_norm   = 1.0 / norm
    inv_m_f    = 1.0 / m
    eta_n      = eta_gf * inv_norm      # combined learning-rate/norm factor
    inv_sqrt2  = 1.0 / math.sqrt(2.0)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for step in trange(N_steps - 1, desc=f"GF d={d} [{dev.type}]"):

        # ── Forward pass ──────────────────────────────────────────────

        # Router discriminant  disc_i = (V[0] - V[1])·x_i / √2
        disc = (X @ (V[0] - V[1])) * inv_sqrt2                     # (n,)
        G0   =  0.5 + a1 * disc                                     # (n,)
        G1   =  0.5 - a1 * disc                                     # (n,)

        # Expert pre-activations  Z[e]_{i,l} = W[e]_{l,·} · x_i
        # Single batched matmul (n×d) @ (d×E*m) = (n, E*m); chunked for OOM safety.
        ZW   = _mm_chunk(X, W.T, chunk_size)                        # (n, E*m)
        Z0   = ZW[:, :m]                                            # (n, m)
        Z1   = ZW[:, m:]                                            # (n, m)

        # Nonlinearity and expert predictions
        phi_Z0 = _phi(Z0)                                           # (n, m)
        phi_Z1 = _phi(Z1)                                           # (n, m)
        Fe0    = a_vec[0] * inv_m_f * phi_Z0.sum(dim=1)            # (n,)
        Fe1    = a_vec[1] * inv_m_f * phi_Z1.sum(dim=1)            # (n,)

        f   = G0 * Fe0 + G1 * Fe1                                   # (n,)
        res = y - f                                                  # (n,)

        # ── Backward pass: W ──────────────────────────────────────────

        # φ'(Z[e]) for both experts
        sp0 = _phi_prime(Z0)                                        # (n, m)
        sp1 = _phi_prime(Z1)                                        # (n, m)

        # Scaled coefficients: (n, 1) broadcast → (n, m)
        cw0 = (a_vec[0] * inv_m_f * G0 * res).unsqueeze(1)        # (n, 1)
        cw1 = (a_vec[1] * inv_m_f * G1 * res).unsqueeze(1)        # (n, 1)

        # Gradient for all experts in one matmul: (E*m, n) @ (n, d)
        grad_acts = torch.cat([cw0 * sp0, cw1 * sp1], dim=1).T    # (E*m, n)
        W = W + eta_n * (grad_acts @ X)                            # (E*m, d)
        W = torch.nn.functional.normalize(W, dim=1)               # sphere

        # ── Backward pass: V ──────────────────────────────────────────

        # coeff_v_i = gp · res_i · (Fe0_i - Fe1_i)
        cv       = gp * res * (Fe0 - Fe1)                          # (n,)
        gv_base  = eta_n * (cv @ X)                                # (d,)

        # Tangent-space update: v_new = v + grad - (grad·v)*v, then normalise
        V0 = V[0] + gv_base  - gv_base.dot(V[0])  * V[0]
        V1 = V[1] - gv_base  - (-gv_base).dot(V[1]) * V[1]
        V  = torch.stack([V0 / V0.norm(), V1 / V1.norm()])        # (E, d)

        # ── Backward pass: a_vec ──────────────────────────────────────
        # Old code: grad_a = η * Σ(G·res·Fe) / (d·m·a)
        # With Fe = a/m·Σσ: → η/(d·m²) · Σ(G·res·Σσ)   [a cancels]
        # Using phi_Z sums directly (= Σ_l σ(Z_{il})) avoids the a factor.
        ga0 = eta_n * (inv_m_f * inv_m_f) * (G0 * res * phi_Z0.sum(dim=1)).sum()
        ga1 = eta_n * (inv_m_f * inv_m_f) * (G1 * res * phi_Z1.sum(dim=1)).sum()
        a_vec = torch.stack([
            (a_vec[0] + ga0).clamp(min=1e-6),
            (a_vec[1] + ga1).clamp(min=1e-6),
        ])

        _record(step + 1, loss_val=float(0.5 * res.pow(2).mean()))

    return {
        't':          t_arr,
        'a':          a_arr,
        'P_self':     P_self_arr,
        'P_cross':    P_cross_arr,
        'Rv_self':    Rv_self_arr,
        'Rv_cross':   Rv_cross_arr,
        'Delta':      P_self_arr - P_cross_arr,
        'Delta_v':    Rv_self_arr - Rv_cross_arr,
        'Pv_self':    Pv_self_arr,
        'Pv_cross':   Pv_cross_arr,
        'train_loss': train_loss_arr,
    }