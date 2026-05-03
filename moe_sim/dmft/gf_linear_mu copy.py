"""
Generating functional (GF) for the linear-router MoE with cluster means.

Data model: x^µ = kappa * U_{c_µ} + xi^µ,  xi ~ N(0, I_d)
Teacher:    y^µ = phi(U_{c_µ} · x^µ)  =  phi(kappa + N(0,1))

Cluster means break the zero-mean parity, restoring:
  - Non-zero Rv_self gradient (router learns cluster direction)
  - Correct P_self growth via the (1/2)*h'(P_self) term
"""

import numpy as np
from scipy.special import erf

phi = lambda z: erf(z / np.sqrt(2.0))


def run_gf_linear_mu(seed=0, alpha_bar=1.0, m=300, d=1200, T_max=8.0, eta_gf=0.1,
                     a0=0.5, P_self0=0.2, Rv_self0=0.05, a1=0.3, kappa=1.0):
    """GF simulation with non-zero cluster means kappa."""
    rng    = np.random.default_rng(seed)
    E      = 2
    n      = int(round(alpha_bar * d))
    gp_val = a1 / np.sqrt(2.0)
    # Normalisation: divide by d (not n) so that the GF matches the DMFT
    # convention.  The DMFT is derived from the gradient flow
    #   dW/dt = (1/d) Σ_μ grad_μ
    # which at n = α·d samples gives a signal proportional to α.  Dividing
    # by n = α·d instead would make the signal independent of α, causing a
    # systematic factor-α error at all α ≠ 1.  (Our earlier Stage 7c
    # validation happened to be at α = 1, where n = d and the bug was silent.)
    norm   = float(d)

    # ── Teacher directions ────────────────────────────────────────────────────
    U = np.zeros((E, d)); U[0, 0] = 1.0; U[1, 1] = 1.0

    # ── Data with cluster means: x = kappa*U_c + xi ──────────────────────────
    cluster = np.arange(n) % E
    X = rng.standard_normal((n, d))
    for c in range(E):
        X[cluster == c] += kappa * U[c]         # ADD cluster mean

    y = np.array([phi(X[i] @ U[cluster[i]]) for i in range(n)])  # phi(kappa + noise)

    # ── Initial expert weights ────────────────────────────────────────────────
    W = []
    for e in range(E):
        rand = rng.standard_normal((m, d))
        rand -= (rand @ U[e, :, None]) * U[e, None, :]
        rand /= np.linalg.norm(rand, axis=1, keepdims=True)
        w  = P_self0 * U[e] + np.sqrt(1 - P_self0**2) * rand
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        W.append(w)

    # ── Initial router weights ────────────────────────────────────────────────
    V = []
    for e in range(E):
        v = rng.standard_normal(d)
        for ep in range(E): v -= (v @ U[ep]) * U[ep]
        for ep in range(e): v -= (v @ V[ep]) * V[ep]
        v /= np.linalg.norm(v)
        v  = Rv_self0 * U[e] + np.sqrt(1 - Rv_self0**2) * v
        v /= np.linalg.norm(v)
        V.append(v)

    a_vec = np.full(E, a0)
    N_steps = int(T_max / eta_gf) + 1

    # ── Storage ───────────────────────────────────────────────────────────────
    t_arr        = np.arange(N_steps) * eta_gf
    a_arr        = np.zeros(N_steps)
    P_self_arr   = np.zeros(N_steps)
    P_cross_arr  = np.zeros(N_steps)
    Rv_self_arr  = np.zeros(N_steps)
    Rv_cross_arr = np.zeros(N_steps)
    Pv_self_arr  = np.zeros(N_steps)
    Pv_cross_arr = np.zeros(N_steps)
    train_loss_arr = np.zeros(N_steps)

    def record(s, loss_val=None):
        a_arr[s]        = float(a_vec.mean())
        P_self_arr[s]   = float(np.mean([np.mean(W[e] @ U[e])   for e in range(E)]))
        P_cross_arr[s]  = float(np.mean([np.mean(W[e] @ U[1-e]) for e in range(E)]))
        Rv_self_arr[s]  = float(np.mean([V[e] @ U[e]             for e in range(E)]))
        Rv_cross_arr[s] = float(np.mean([V[0] @ U[1], V[1] @ U[0]]))
        Pv_self_arr[s]  = float(np.mean([np.mean(W[e] @ V[e])   for e in range(E)]))
        Pv_cross_arr[s] = float(np.mean([np.mean(W[e] @ V[1-e]) for e in range(E)]))
        if loss_val is not None:
            train_loss_arr[s] = loss_val

    # Compute initial training loss before any update
    disc0 = (X @ V[0] - X @ V[1]) / np.sqrt(2.0)
    G0    = [0.5 + a1*disc0, 0.5 - a1*disc0]
    Z0    = [X @ W[e].T for e in range(E)]
    Fe0   = [(a_vec[e]/m) * phi(Z0[e]).sum(axis=1) for e in range(E)]
    f0_   = G0[0]*Fe0[0] + G0[1]*Fe0[1]
    res0_ = y - f0_
    record(0, loss_val=float(np.mean(res0_**2)/2))

    for step in range(N_steps - 1):
        # Forward pass (CLUSTERED routing for both prediction and gradients)
        disc = (X @ V[0] - X @ V[1]) / np.sqrt(2.0)
        # Unclipped linear router — matches the DMFT derivation.
        # The DMFT uses g_e = a0 + a1·disc without constraints, so the GF must do
        # the same. Clipping to [0,1] reduces the gradient by ~5-9% (fraction of
        # data with |disc| > 0.5/a1 ≈ 1.67σ), creating a systematic signal deficit
        # that compounds via positive P_self feedback into a ~50% gap over T=8.
        G    = [0.5 + a1*disc, 0.5 - a1*disc]
        Z    = [X @ W[e].T for e in range(E)]
        Fe   = [(a_vec[e]/m) * phi(Z[e]).sum(axis=1) for e in range(E)]
        f    = G[0]*Fe[0] + G[1]*Fe[1]     # ← routing IS in the forward pass
        res  = y - f

        # Expert gradients (with routing weight G[e])
        for e in range(E):
            sig_p    = np.sqrt(2/np.pi) * np.exp(-0.5*Z[e]**2)   # (n, m)
            coeff_w  = (a_vec[e] / m) * G[e] * res   # (n,) — 1/m from f_e=(1/m)Σσ (paper eq.4)
            grad_w   = coeff_w[:, None] * sig_p                   # (n, m)
            dW       = eta_gf * (grad_w.T @ X) / norm               # (m, d)
            W[e]    += dW
            norms    = np.linalg.norm(W[e], axis=1, keepdims=True)
            W[e]    /= norms

        # Router gradient: d/dV_e f = gp*(G[0]-G[1]) sign * (Fe[0]-Fe[1]) * x
        # For V_0: ∂f/∂(V_0·x) = +gp*(Fe[0]-Fe[1])
        # For V_1: ∂f/∂(V_1·x) = -gp*(Fe[0]-Fe[1])
        diff_F = Fe[0] - Fe[1]   # (n,)
        for e in range(E):
            sign_e   = 1.0 if e == 0 else -1.0
            coeff_v  = gp_val * sign_e * res * diff_F    # (n,)
            grad_v   = eta_gf * (coeff_v @ X) / norm        # (d,)
            proj     = np.dot(grad_v, V[e])
            V[e]    += grad_v - proj*V[e]
            V[e]    /= np.linalg.norm(V[e])

        # a_bar gradient (1/m from f_e=(1/m)*sum_l, same derivation as coeff_w)
        for e in range(E):
            grad_a  = eta_gf * np.sum(G[e] * res * Fe[e]) / (norm * m * a_vec[e])
            a_vec[e] += grad_a
            a_vec[e]  = np.clip(a_vec[e], 1e-6, None)

        # Compute training loss at end of this step (BEFORE next step's record)
        disc_  = (X @ V[0] - X @ V[1]) / np.sqrt(2.0)
        G_    = [0.5 + a1*disc_, 0.5 - a1*disc_]
        Z_    = [X @ W[e].T for e in range(E)]
        Fe_   = [(a_vec[e]/m) * phi(Z_[e]).sum(axis=1) for e in range(E)]
        f__   = G_[0]*Fe_[0] + G_[1]*Fe_[1]
        res__ = y - f__
        record(step + 1, loss_val=float(np.mean(res__**2)/2))

    return {
        't': t_arr, 'a': a_arr,
        'P_self': P_self_arr, 'P_cross': P_cross_arr,
        'Rv_self': Rv_self_arr, 'Rv_cross': Rv_cross_arr,
        'Delta':   P_self_arr  - P_cross_arr,
        'Delta_v': Rv_self_arr - Rv_cross_arr,
        'Pv_self': Pv_self_arr, 'Pv_cross': Pv_cross_arr,
        'train_loss': train_loss_arr,
    }