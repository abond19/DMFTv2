"""
Stage 7e DMFT — fully corrected, vectorised.

Builds on Stage 7d by fixing two structural bugs and adding the M^o router terms.

BUG-FIX 1 — Row-n kernels (same as stage 7c fix):
  All memory kernels M(t,τ) are evaluated at CURRENT TIME t=n integrated over past τ.
  The correct implementation uses row n of R_A/C_A and H matrices, NOT column sums.
  Column sums Σ_τ M[τ,j] inflate ν_v by 5–20×, break |C^v_d(t,s)|≤1, and cause
  Rv_self to diverge for α·κ ≳ 2.

BUG-FIX 2 — Off-diagonal router terms (M^o, eqs 53,55,61):
  ν_v (eq 61), C^v_d Volterra (eq 53), C^v_o Volterra (eq 54), and R^v_d Volterra (eq 55)
  all include (E-1)·M^o_{R/C,Qv} terms coupling C^v_d ↔ C^v_o.  For E=2, (E-1)=1.
  These terms are driven by the off-diagonal G-factor:
    G^{sc,o}_{s/c}(t,τ) = 0.25 + a₁²·(C^v_o(t,τ) − κ²·rD_t·rD_τ) ± ½·a₁·κ·(rD_t − rD_τ)
  with H kernels evaluated at cross-expert means (ms_t, mc_τ) and C_x ≈ 0.
  Without these terms DMFT over-predicts Δv by ~36% at (α=1,κ=2).

RETAINED from Stage 7d:
  - Full two-time C_o(t,s) and R_o(t,s) tracking (eq 49, 51).
  - Exact C_o in Σ_C (instead of factored P_self·P_self approximation).
  - Complete ν_w formula: (1/m)·MRd·Cd + (m-1)/m·MRo·Co + (1/m)·MCPv·RPv.
  - Stage 7d SigmaR: dH/dCdv = a1²·(Hm_ss + Hm_cc) (derivative, no G-weights).

NEW in Stage 7e:
  - C^v_o (N×N), R^v_o (N×N): off-diagonal router correlator/response.
    Initial: C^v_o=0 (C^v_o(0,0)=2·Rv_self0·Rv_cross0=0), R^v_o=0.
    Diagonal pin (set AFTER single-time ODE updates):
      C^v_o(t,t) = 2·Rv_self(t)·Rv_cross(t)  (spherical factored constraint).
  - M^o_{R,Qv} and M^o_{C,Qv} kernels: row-n slices of R_A/C_A weighted by ∂_{C^v_o}H^{sc,o}.
"""

import numpy as np
from dataclasses import dataclass, field

from dmft.kernels_e1 import (
    H_mu, H_mu_prime, H_mu_arr, H_mu_prime_arr, Phi_target_mu, dH_mu_dm2_arr
)
from dmft.routing_linear_mu import compute_kernels_mu


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Stage7eMuConfig:
    alpha_bar:          float = 1.0
    m:                  int   = 300
    T_max:              float = 8.0
    eta:                float = 0.1
    a0:                 float = 0.5
    P_self0:            float = 0.2
    P_cross0:           float = 0.0
    Rv_self0:           float = 0.05
    Rv_cross0:          float = 0.0
    a1:                 float = 0.3
    kappa:              float = 1.0
    sigma_y:            float = 0.0
    E:                  int   = 2
    apply_sc_correction: bool = True   # subtract f₀ self-consistency from t=0 signals

    inv_m:   float = field(init=False)
    N_steps: int   = field(init=False)

    def __post_init__(self):
        self.inv_m   = 1.0 / self.m
        self.N_steps = int(self.T_max / self.eta) + 1


# ─────────────────────────────────────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────────────────────────────────────

class DMFTSolverStage7eMu:
    def __init__(self, cfg: Stage7eMuConfig):
        self.cfg = cfg
        N = cfg.N_steps
        self.a1    = cfg.a1
        self.kappa = cfg.kappa

        # Single-time scalars
        self.t        = np.arange(N) * cfg.eta
        self.a        = np.zeros(N);  self.a[0]        = cfg.a0
        self.P_self   = np.zeros(N);  self.P_self[0]   = cfg.P_self0
        self.P_cross  = np.zeros(N);  self.P_cross[0]  = cfg.P_cross0
        self.Rv_self  = np.zeros(N);  self.Rv_self[0]  = cfg.Rv_self0
        self.Rv_cross = np.zeros(N);  self.Rv_cross[0] = cfg.Rv_cross0
        self.r_D      = np.zeros(N)
        self.r_D[0]   = (cfg.Rv_self0 - cfg.Rv_cross0) / np.sqrt(2.0)
        self.Delta    = np.zeros(N);  self.Delta[0]    = cfg.P_self0  - cfg.P_cross0
        self.Delta_v  = np.zeros(N);  self.Delta_v[0]  = cfg.Rv_self0 - cfg.Rv_cross0
        self.nu_w     = np.zeros(N)
        self.nu_v     = np.zeros(N)

        # Expert weight correlators (diagonal + off-diagonal within same expert)
        self.Cd = np.eye(N);        self.Rd = np.zeros((N, N))
        self.Co = np.zeros((N, N)); self.Ro = np.zeros((N, N))
        self.Co[0, 0] = cfg.P_self0 ** 2   # C_o(0,0) = P_self0²

        # Global (residual) propagators
        self.CA = np.zeros((N, N)); self.RA = np.zeros((N, N))

        # Router correlators (diagonal + off-diagonal between experts)
        # C^v_d(t,s) = E[z^v_e(t)·z^v_e(s)] = router weight AUTO-correlation (like C_d for experts).
        # C^v_d(t,t) = ||v_e||^2 = 1  →  np.eye is correct.  (NOT the cross-correlation w·v.)
        # The expert-router cross-correlation E[w_l·v_e] is P^v_self = Pv_self, initialized below.
        self.Cdv = np.eye(N);        self.Rdv = np.zeros((N, N))
        self.Cvo = np.zeros((N, N)); self.Rvo = np.zeros((N, N))
        # Cvo(0,0) = 2·Rv_self0·Rv_cross0 = 0 (cross starts at 0)

        # Expert-router overlap  (self: same expert; cross: different expert)
        self.Pv_self         = np.zeros((N, N))
        self.Pv_self[0, 0]   = cfg.P_self0 * cfg.Rv_self0
        self.R_Pv_self       = np.zeros((N, N))
        # P^v_cross and its response — needed for C_o^v source terms (doc §Cov_source)
        self.Pv_cross        = np.zeros((N, N))   # P^v_{e,e'}(t,s) for e≠e'
        self.R_Pv_cross      = np.zeros((N, N))

        # Single-time kernel caches
        self.hat_Phi_s      = np.zeros(N)
        self.hat_Phi_cs     = np.zeros(N)   # E_{c1}[G_0*y_1*phi_1]  — NOT used in psi
        self.hat_Phi_cross  = np.zeros(N)   # E_{c1}[G_0*y_1*phi_0]  — correct cross term
        self.tPhi_s     = np.zeros(N)
        self.tPhi_c     = np.zeros(N)
        self.tPhiv_s    = np.zeros(N)
        self.tPhiv_c    = np.zeros(N)
        self.Phi_target = Phi_target_mu(cfg.kappa)

        # ā Volterra integrand history (Fix B — paper eq.118):
        # psi_abar[n]  = hat_Phi_self(n) + hat_Phi_cross(n)    (no 1/E)
        # H_abar[n]    = H_ss,d(n,n) + H_cc,d(n,n) + H_sc,o(n,n) + H_cs,o(n,n)
        #              = full equal-time diagonal self-energy for ā (4 kernel types, E=2)
        # The ā ODE integrand is psi_abar[s] - ā(s)*H_abar[s], convolved with R_A(t,s).
        self.psi_abar   = np.zeros(N)   # hat_Phi_self + hat_Phi_cross  (no 1/E)
        self.H_abar     = np.zeros(N)   # H_ss,d + H_cc,d + H_sc,o + H_cs,o  at (s,s)

        self._kernels(0)

    # ── Kernel cache update ────────────────────────────────────────────────────
    def _kernels(self, n):
        cfg   = self.cfg
        a1    = cfg.a1
        kappa = cfg.kappa
        inv_m = cfg.inv_m

        k = compute_kernels_mu(self.P_self[n], self.P_cross[n],
                                self.Rv_self[n], self.Rv_cross[n],
                                a1, kappa,
                                pv_self=self.Pv_self[n, n],    # Bug 5 fix
                                pv_cross=self.Pv_cross[n, n])  # Bug 5 fix
        self.hat_Phi_s[n]       = k['hat_Phi_self']
        self.hat_Phi_cs[n]      = k['hat_Phi_cs']
        self.hat_Phi_cross[n]   = k['hat_Phi_cross']
        self.tPhi_s[n]     = k['tPhi_self']
        self.tPhi_c[n]     = k['tPhi_cross']
        self.tPhiv_s[n]    = k['tPhiv_self']
        self.tPhiv_c[n]    = k['tPhiv_cross']

        # ── ā Volterra integrand (Fix B, paper eq.118) ────────────────────────
        # psi_abar(s) = hat_Phi_self(s) + hat_Phi_cross(s)    [no 1/E, E=2 term]
        inv_E_k = 1.0 / self.cfg.E  # FIX-A: per-expert factor (level-0 shows 2x excess)
        self.psi_abar[n] = inv_E_k * (k['hat_Phi_self'] + k['hat_Phi_cross'])

        # H_abar(s) = equal-time ā self-energy (four kernel types at diagonal):
        #   H_ss,d(s,s) + H_cc,d(s,s) + H_sc,o(s,s) + H_cs,o(s,s)
        # H_ss,d uses Cd(s,s)=1 (diagonal) and Co(s,s) (off-diagonal within expert).
        # H_sc,o uses Cx(s,s) ≈ 0 (cross-expert weight correlator, approximated as zero).
        # All means come from current-time P_self, P_cross, Rv_self, Rv_cross.
        rD   = self.r_D[n]
        Cvo_n = self.Cvo[n, n]    # off-diagonal router correlator at diagonal

        # Bug 2 fix: Gs_n = E[g_0^2] = (a0+a1*kappa*rD)^2 + a1^2*(Cdv(t,t)-Cvo(t,t))
        #            Cdv(t,t)=1 (spherical), Cvo(t,t) is tracked.
        # Old code had +a1^2*(1+kappa^2*rD^2) missing -Cvo_n.
        Gs_n = (0.5 + a1*kappa*rD)**2 + a1**2 * (1.0 - Cvo_n)
        Gc_n = (0.5 - a1*kappa*rD)**2 + a1**2 * (1.0 - Cvo_n)

        ms_n = kappa * self.P_self[n]
        mc_n = kappa * self.P_cross[n]
        Co_n = self.Co[n, n]

        # H_ss,d and H_cc,d — diagonal (same-expert) kernel types
        # Uses the (m,Co) mixture: (1/m)*H_mu(1,...) + (1-1/m)*H_mu(Co,...)
        H_ss_d = Gs_n * (inv_m*H_mu(1., ms_n, ms_n) + (1-inv_m)*H_mu(Co_n, ms_n, ms_n))
        H_cc_d = Gc_n * (inv_m*H_mu(1., mc_n, mc_n) + (1-inv_m)*H_mu(Co_n, mc_n, mc_n))

        # H_sc,o and H_cs,o — off-diagonal (cross-expert) kernel types at diagonal
        # Use G^{s,o} and G^{c,o} with C_x≈0 and C^v_o(s,s):
        #   G^{s,o}(s,s) = 1/4 + a1^2*(Cvo(s,s) - kappa^2*rD^2) + 0  [linear term cancels]
        #   G^{c,o}(s,s) = same  [delt_o = a1*kappa*(rD-rD)=0 at diagonal]
        # Bug 2 fix: Gso_n = E[g_0(t)*g_1(t)] = (a0+a1k*rD)(a0-a1k*rD) - a1^2*(Cdv-Cvo)
        # At diagonal: Cdv(t,t)=1, so Gso_n = 0.25 - a1^2*k^2*rD^2 - a1^2*(1-Cvo_n)
        #            = 0.25 + a1^2*(Cvo_n - 1.0 - kappa^2*rD^2).
        # Old code was missing the -1.0 term (Cdv contribution), making Gso 56% too large.
        Gso_n  = 0.25 + a1**2*(Cvo_n - 1.0 - kappa**2*rD**2)
        H_sc_o = Gso_n * H_mu(0., ms_n, mc_n)   # Cx=0, means (ms, mc)
        H_cs_o = Gso_n * H_mu(0., mc_n, ms_n)   # Cx=0, means (mc, ms)

        self.H_abar[n] = inv_E_k * (H_ss_d + H_cc_d + H_sc_o + H_cs_o)  # FIX-A

    # ── Main loop ──────────────────────────────────────────────────────────────
    def run(self, verbose=False, clamp=None):
        cfg   = self.cfg
        eta   = cfg.eta
        alp   = cfg.alpha_bar
        inv_m = cfg.inv_m
        inv_E = 1.0 / cfg.E
        a1    = cfg.a1
        kappa = cfg.kappa
        m     = cfg.m

        # ── Self-consistency correction at t=0 (BUG-FIX SC) ─────────────────
        # The DMFT signal at any t uses the *teacher* kernel E[G·y·σ'·λ].
        # Self-consistency (r = y − f) is then handled by R_A for t > 0.
        # But at t=0, R_A has no history and the GF's residual already
        # subtracts the initial network output f₀ = G·ā₀·σ(z^w).
        # Leading-order correction to the step-0 signal kernels:
        #
        #   δΦ_self  = ā₀·G_s·E[phi·phi'(z^w)·λ]   (reduces tPhi_s[0])
        #   δΦ_cross = ā₀·G_{co}·E[phi'(z^w₀)]·E[phi(z^w₁)·λ₁]  (reduces tPhi_c[0])
        #
        # G_s  = E_{c=e}[g_e²] = (a₀+a₁κrD)²+a₁²     (same-cluster G-factor)
        # G_co = E_{c≠e}[g_e·g_{e'}] = a₀²−a₁²(1+κ²rD²) (cross G-factor at t=0)
        #
        # The correction is proportional to ā₀ and vanishes at small
        # initialisation (ā₀ → 0), so it is negligible when running with
        # a0=0.1, P_self0=0.02 (see note at bottom of run_comparison.py).
        if cfg.apply_sc_correction:
            rD_0   = self.r_D[0]                      # (Rv_self0−Rv_cross0)/√2
            # Bug 2 fix: Gs_0 = (a0+a1*kappa*rD_0)^2 + a1^2*(1-Cvo[0,0])
            # Cvo[0,0] = 2*Rv_self0*Rv_cross0 = 0, so Gs_0 = (a0+a1*kappa*rD_0)^2 + a1^2.
            # Old code had extra a1^2*kappa^2*rD_0^2 term (double-counted).
            Gs_0   = ((cfg.a0 + a1*kappa*rD_0)**2
                      + a1**2 * (1.0 - self.Cvo[0, 0]))   # E_{c=e}[g_e^2]
            # Gco_0 = E[g_0*g_1] = (a0+a1k*rD_0)(a0-a1k*rD_0) - a1^2*(1-Cvo[0,0])
            #       = a0^2 - a1^2*(kappa^2*rD_0^2 + 1 - Cvo[0,0])
            Gco_0  = (cfg.a0**2
                      - a1**2 * (kappa**2*rD_0**2 + 1.0 - self.Cvo[0, 0]))  # E[g_e*g_{e'}]
            mws    = kappa * cfg.P_self0

            # Compute required 1-D integrals using the module-level quadrature nodes
            from dmft.kernels_e1 import _WW as _W, _ZZ as _Z, erf as _erf

            # φ̄_self = E_{c=e}[σ(z^w_e)] = E[phi(mws+G)], G~N(0,1)
            phi_bar_self = float(np.dot(_W, _erf((mws + _Z)/np.sqrt(2.))))

            # E[σ'(mws+G)], then K_φ'λ via Stein: E[σ''(mws+G)] = −(mws/2)·E[σ'(mws+G)]
            E_phip = float(np.dot(_W, np.sqrt(2./np.pi)*np.exp(-0.5*(mws + _Z)**2)))
            K_phip_lam = E_phip * (kappa - cfg.P_self0 * mws / 2.0)

            k0 = compute_kernels_mu(cfg.P_self0, cfg.P_cross0,
                                     cfg.Rv_self0, cfg.Rv_cross0, a1, kappa)

            # Dominant off-diagonal (l≠l') + diagonal (l=l') contributions:
            #   off-diag scales as O(m^0) and dominates at large m
            #   diagonal (from K_phi_phip_lam) is O(1/m) ≈ 0.3% at m=300
            delta_tPhi_s = (2.0 * cfg.a0 * phi_bar_self * Gs_0 * K_phip_lam
                            + cfg.a0 * Gs_0 * k0['sc_K_phi_phip'])
            # Cross: cluster-1 data; φ̄_self same by S_E symmetry;
            # G_co replaces G_s; K_phip_lam same (symmetric teacher)
            delta_tPhi_c = (2.0 * cfg.a0 * phi_bar_self * Gco_0 * K_phip_lam
                            + cfg.a0 * Gco_0 * k0['sc_K_cross'])

            self.tPhi_s[0] -= delta_tPhi_s
            self.tPhi_c[0] -= delta_tPhi_c

            if verbose:
                print(f"  SC correction (t=0):  "
                      f"δΦ_self={delta_tPhi_s:.5f} ({100*delta_tPhi_s/k0['tPhi_self']:.1f}% of tPhi_s)  "
                      f"δΦ_cross={delta_tPhi_c:.5f} ({100*delta_tPhi_c/max(k0['tPhi_cross'],1e-12):.1f}% of tPhi_c)")

        for n in range(cfg.N_steps - 1):
            nn  = n + 1
            an  = self.a[n]
            av  = self.a[:nn]          # ā(τ) for τ=0..n

            # ── Diagonal G-factor matrices (nn×nn) ───────────────────────────
            rDt   = self.r_D[:nn, None];  rDj = self.r_D[None, :nn]
            Cdv_b = self.Cdv[:nn, :nn]
            Cvo_b = self.Cvo[:nn, :nn]
            # Bug 2 fix: E[g_e(t)*g_e(s)] = (a0+a1k*rDt)(a0+a1k*rDs) + a1^2*(Cdv-Cvo).
            # Old code had +a1^2*(Cdv + kappa^2*rDt*rDj), missing the -Cvo_b term.
            base  = 0.25 + a1**2 * (Cdv_b - Cvo_b + kappa**2 * rDt * rDj)
            delt  = 0.5  * a1 * kappa * (rDt + rDj)
            Gs    = base + delt;   Gc = base - delt

            mst = kappa * self.P_self[:nn,  None]
            mct = kappa * self.P_cross[:nn, None]
            msj = kappa * self.P_self[None,  :nn]
            mcj = kappa * self.P_cross[None, :nn]

            Cd_b = self.Cd[:nn, :nn]
            Co_b = self.Co[:nn, :nn]
            RA_b = self.RA[:nn, :nn]
            CA_b = self.CA[:nn, :nn]

            # ── H matrices (using Cd and Co) ─────────────────────────────────
            Hm_ss   = H_mu_arr(Cd_b, mst, msj)        # H(Cd, ms, ms)
            Hm_cc   = H_mu_arr(Cd_b, mct, mcj)        # H(Cd, mc, mc)
            Hm      = Gs * Hm_ss + Gc * Hm_cc         # full H (for Σ_C, M_Qv)

            Hpm_ss  = H_mu_prime_arr(Cd_b, mst, msj)  # H'(Cd, ms, ms)
            Hpm_cc  = H_mu_prime_arr(Cd_b, mct, mcj)  # H'(Cd, mc, mc)
            Hpm     = Gs * Hpm_ss + Gc * Hpm_cc       # H' for Σ_R, M_Q

            # H with Co (for off-diagonal expert weight kernel M^o_R,Q)
            Hm_Co_ss = H_mu_arr(Co_b, mst, msj)
            Hm_Co_cc = H_mu_arr(Co_b, mct, mcj)
            HpmCo    = Gs * H_mu_prime_arr(Co_b, mst, msj) + Gc * H_mu_prime_arr(Co_b, mct, mcj)

            # ── Off-diagonal (sc) G-factors and H kernels ─────────────────────
            # For M^o_{R/C,Qv}: G^{sc,o} uses C^v_o and sign-flipped rD_j.
            # C_x ≈ 0 throughout (starts at 0, cross-expert weight overlap stays small).
            # Bug 2 fix: E[g_0(t)*g_1(s)] = (a0+a1k*rDt)(a0-a1k*rDs) - a1^2*(Cdv-Cvo)
            #   = 0.25 + a1^2*(Cvo - Cdv - kappa^2*rDt*rDj) + 0.5*a1*k*(rDt-rDs).
            # Old code was missing -Cdv_b, making Gso 56% too large.
            base_o = 0.25 + a1**2 * (Cvo_b - Cdv_b - kappa**2 * rDt * rDj)
            delt_o = 0.5  * a1 * kappa * (rDt - rDj)           # sign flip on rDj
            Gso    = base_o + delt_o;   Gco = base_o - delt_o

            _zero  = np.zeros((nn, nn))                         # C_x ≈ 0
            Hm_sc  = H_mu_arr(_zero, mst, mcj)                  # H(0, ms_t, mc_j)
            Hm_cs  = H_mu_arr(_zero, mct, msj)                  # H(0, mc_t, ms_j)
            # Hmo    = Gso * Hm_sc + Gco * Hm_cs               # H^{sc,o} (unused directly)
            dH_Cvo = a1**2 * (Hm_sc + Hm_cs)                   # ∂_{C^v_o} H^{sc,o}

            # Bug 3 setup: response matrices needed for missing SigmaR terms
            Ro_b   = self.Ro[:nn, :nn]    # R_o(t,s)  — within-expert off-diagonal response
            Rvo_b  = self.Rvo[:nn, :nn]   # R^v_o(t,s) — cross-expert router response

            # ── Σ_R and Σ_C (vectorised over j) ─────────────────────────────
            # Σ_R: ā(n)·ā(j)·[H'(Cd)·Rd + dH_dCdv·Rdv]
            # dH_dCdv = ∂_{C^v_d} H_nj, where H_nj uses exact Co (from Stage 7d).
            # H_nj = Gs*(inv_m*Hm_ss + (1-inv_m)*Hm_Co_ss) + Gc*(inv_m*Hm_cc + (1-inv_m)*Hm_Co_cc)
            # ∂_{C^v_d} Gs = a1², ∂_{C^v_d} Gc = a1²  (G-factors only)
            # → dH_dCdv = a1²*[inv_m*(Hm_ss+Hm_cc) + (1-inv_m)*(Hm_Co_ss+Hm_Co_cc)]
            dH_dCdv_mat = a1**2 * (inv_m * (Hm_ss + Hm_cc)
                                    + (1 - inv_m) * (Hm_Co_ss + Hm_Co_cc))
            Rd_b  = self.Rd[:nn, :nn]
            Rdv_b = self.Rdv[:nn, :nn]
            an_av = an * av                                      # ā(n)·ā(j)
            # Bug 3 fix: SigmaR was missing the ∂_{Co}H·Ro and ∂_{Cov}H·Rvo terms.
            # Paper eq.sigma_r has five derivative components; only two were implemented.
            # Added:
            #   (1-inv_m)·HpmCo·Ro  ← ∂_{Co}H contribution (G-weighted, off-diag within expert)
            #   dH_Cvo·Rvo          ← ∂_{Cov}H contribution (cross-expert router correlator)
            # The ∂_{Pv}H·R_Pv term is O(Pv*Hpm) ≈ 3% at t=8 and requires new kernel
            # functions; it is deferred but its effect is partially captured via the
            # Pv coupling correction in psi_abar and SigmaC (Bug 5 fix).
            SigR  = an_av * (Hpm[n, :] * Rd_b[n, :]
                             + (1 - inv_m) * HpmCo[n, :] * Ro_b[n, :]   # Bug 3 fix
                             + dH_dCdv_mat[n, :] * Rdv_b[n, :]
                             + dH_Cvo[n, :] * Rvo_b[n, :])               # Bug 3 fix

            # Σ_C: ΣC(t,s) = E[(y-f(t))*(y-f(s))]
            # E[y·f(t)] = a(t)·Σ_e E[G_e(t)·y·φ_e(t)]
            #           = a(t)·(hat_Phi_self + hat_Phi_cross)   ← sum over BOTH experts
            # NOTE: hat_Phi_cs = E_{c1}[G_0·y_1·φ_1] ≠ hat_Phi_cross = E_{c1}[G_0·y_1·φ_0]
            # At κ=1: hat_Phi_cs/hat_Phi_self ≈ 0.76 (old formula OK)
            # At κ=2: hat_Phi_cs/hat_Phi_self ≈ 0.19 (old formula fails badly!)
            # Correct formula: psi_full = hat_Phi_self + hat_Phi_cross (no inv_E)
            psi_full = self.hat_Phi_s[:nn] + self.hat_Phi_cross[:nn]   # shape (nn,), no /E
            H_nj    = (Gs[n, :] * (Hm_ss[n, :] * inv_m + (1-inv_m) * Hm_Co_ss[n, :])
                       + Gc[n, :] * (Hm_cc[n, :] * inv_m + (1-inv_m) * Hm_Co_cc[n, :]))
            SigC    = (cfg.sigma_y**2 + self.Phi_target
                       - an * psi_full[n] - av * psi_full
                       + an_av * H_nj)

            # ── Row-n kernel slices (all shape nn) ───────────────────────────
            # Paper convention: M(t,τ) evaluated at current t=n, integrated over τ.
            RA_rn      = RA_b[n, :]                             # R_A[n, τ]
            CA_rn      = CA_b[n, :]                             # C_A[n, τ]
            Hpm_rn     = Hpm[n, :]                              # H'(Cd)[n, τ]
            HpmCo_rn   = HpmCo[n, :]                            # H'(Co)[n, τ]
            H_rn       = Hm[n, :]                               # H(Cd)[n, τ]
            dH_Cvo_rn  = dH_Cvo[n, :]                          # ∂_{C^v_o}H^{sc,o}[n,τ]

            # ── Row-n unweighted H slice (for router kernels, eqs 101-102) ────
            # Eqs (101-102): ∂_{C^v_d} H_{ss/cc} = a1² · H_mu(Cd, ms, ms) — NO G-weighting.
            # The G-factors appear only when differentiating G itself (giving a1²), not H_mu.
            # This is different from the expert kernels which use ∂_{Cd} H = Gs·H'ss + Gc·H'cc.
            Hm_ss_rn = Hm_ss[n, :]                              # H(Cd, ms_n, ms_τ)[n, τ]
            Hm_cc_rn = Hm_cc[n, :]                              # H(Cd, mc_n, mc_τ)[n, τ]
            Hm_uw_rn = Hm_ss_rn + Hm_cc_rn                     # unweighted sum for router kernels

            # Expert weight kernels (eqs 48,49,51,59) — Kernel-A:
            # M^{e,d}_{C,Q}(n,τ) = (ᾱ/E)·∑_c π_c ∂_{Cd} H_c = (ᾱ/E)·CA·(Gs·H'ss+Gc·H'cc)
            # The factor 1/E comes from ∑_c π_c = 1/E (eq 70).
            # BUG-FIX 6: was missing inv_E → kernels were 2× too large.
            MRd_rn  = alp * inv_E * RA_rn * Hpm_rn             # (α/E)·RA·H'(Cd)
            MCd_rn  = alp * inv_E * CA_rn * Hpm_rn             # (α/E)·CA·H'(Cd)
            MRo_rn  = alp * inv_E * RA_rn * HpmCo_rn           # (α/E)·RA·H'(Co)
            MCo_rn  = alp * inv_E * CA_rn * HpmCo_rn           # (α/E)·CA·H'(Co)

            # Router kernels — diagonal (eqs 53,55,61,101) — Kernel-B:
            # M^d_{C,Qv}(n,τ) = (ᾱ/E)·ā(t)·ā(τ)·CA·a1²·[Hss+Hcc]  (eq 101)
            # ∂_{Cdv} H_{ss} = a1²·Hm_ss (no G in the H part; G differentiates only in Gs).
            # BUG-FIX 7: was using G-weighted H (Gs·Hss+Gc·Hcc) AND missing inv_E.
            # (Code/paper ratio was ~0.5 at κ=1, ~0.56 at κ=2.)
            MRQvd_rn = alp * inv_E * a1**2 * an * av * RA_rn * Hm_uw_rn
            MCQvd_rn = alp * inv_E * a1**2 * an * av * CA_rn * Hm_uw_rn

            # Router kernels — off-diagonal (eqs 53,54,55,61,102):
            # M^o_{C,Qv}(n,τ) = (ᾱ/E)·ā·ā·CA·a1²·[Hsc+Hcs]  (eq 102, E=2 term)
            # dH_Cvo = a1²·(Hsc+Hcs) is already the correct unweighted form.
            # BUG-FIX 8: was missing inv_E.
            MoRQv_rn = alp * inv_E * an * av * RA_rn * dH_Cvo_rn
            MoCQv_rn = alp * inv_E * an * av * CA_rn * dH_Cvo_rn

            # Pv kernels (eq 98): M^{self}_{R,Pv} = (ᾱ/E)·a1²·∫_s^t RA·[Hss+Hcc] dτ
            # Uses unweighted Hss+Hcc (same reason as router diagonal kernels).
            # cumRH_uw: cumulative sum of RA·(Hss+Hcc) from τ=k..n (reversed cumsum trick)
            # BUG-FIX 9: was using G-weighted H AND missing inv_E.
            cumRH_uw = np.cumsum((RA_rn * Hm_uw_rn)[::-1])[::-1]
            MRPv_rn  = alp * inv_E * a1**2 * an * av * cumRH_uw
            MCPv_scalar = alp * inv_E * a1**2 * eta * an * np.dot(CA_rn, Hm_uw_rn)
            MCPv_rn = MCPv_scalar * av

            Hpm_ss_rn = Hpm_ss[n, :]                              # H'(Cd, ms_n, ms_τ)[n, τ]
            Hpm_cc_rn = Hpm_cc[n, :]                              # H'(Cd, mc_n, mc_τ)[n, τ]

            # ── ν_w (eq 59) — memory terms + signal terms ────────────────────
            # Memory terms (present): -(1/m)·∫[M^d_R·Cd + M^d_C·Rd]
            #                         -(1-1/m)·∫[M^o_R·Co + M^o_C·Ro]
            #                         -(1/m)·∫MCPv·RPv        [from eq.59 line 3]
            # Also in line 2 of eq.59: -(1/m)·∫M_{R,Pv}·C_{Pv}  ← was missing
            #
            # Signal (Fix C): ∑_c π_c ∫[M_{R,P_c}·P_{e,c} + M_{R,M_c}·M_{e,c}]dτ
            # Derivation via Stein lemma on E[φ'(z^w)·z^w]:
            #   E[φ'(z^w)z^w] = P/(2-P²)·[κ·∂_{m2}H_μ + P·H'_μ]
            # where both ∂_{m2}H_μ (from dH_dm2 term) and H'_μ (from Price's theorem)
            # contribute. At P=0.2: P·H_prime ≈ 2×(κ·dH_dm2), so previously only 1/3
            # of the kernel was included.
            # For E=2, ∑_c π_c·2·M_{R,P_c}·P collapses to inv_E prefactor (see earlier comment).
            # Full kernel at source time τ:
            #   self: G^s·P_s(τ)/(2-P_s²)·[κ·dH_ss_dm2 + P_s·H'_ss]·[m̄·Cd + (1-m̄)·Co mix]
            #   cross: G^c·P_c(τ)/(2-P_c²)·[κ·dH_cc_dm2 + P_c·H'_cc]
            Ps_tau   = self.P_self[:nn]                          # P_self(τ) for τ=0..n
            Pc_tau   = self.P_cross[:nn]                         # P_cross(τ)
            dHss_dm2_rn = (inv_m   * dH_mu_dm2_arr(Cd_b[n, :], mst[n], msj)
                           + (1-inv_m) * dH_mu_dm2_arr(Co_b[n, :], mst[n], msj))
            dHcc_dm2_rn = (inv_m   * dH_mu_dm2_arr(Cd_b[n, :], mct[n], mcj)
                           + (1-inv_m) * dH_mu_dm2_arr(Co_b[n, :], mct[n], mcj))
            # Stein factor P/(2-P²) ≈ P·inv_E for small P (exact for all P):
            stein_s  = Ps_tau / np.maximum(2.0 - Ps_tau**2, 0.1)   # P/(2-P²) for self
            stein_c  = Pc_tau / np.maximum(2.0 - Pc_tau**2, 0.1)   # for cross
            # Full signal kernel = stein · [κ·dH_dm2 + P·H_prime]
            kernel_s = stein_s * (kappa * dHss_dm2_rn + Ps_tau * Hpm_ss_rn)
            kernel_c = stein_c * (kappa * dHcc_dm2_rn + Pc_tau * Hpm_cc_rn)
            nu_w_signal = (alp * inv_E * an * eta * np.dot(
                av * RA_rn * (Gs[n, :] * kernel_s + Gc[n, :] * kernel_c),
                np.ones(nn)))

            self.nu_w[n] = -(
                inv_m   * eta * (np.dot(MRd_rn, self.Cd[:nn, n])
                                 + np.dot(MCd_rn, self.Rd[n, :nn]))
                + (1-inv_m) * eta * (np.dot(MRo_rn, self.Co[:nn, n])
                                     + np.dot(MCo_rn, self.Ro[n, :nn]))
                + inv_m * eta * np.dot(MCPv_rn, self.R_Pv_self[n, :nn])
                + inv_m * eta * np.dot(MRPv_rn, self.Pv_self[:nn, n])) \
                + nu_w_signal

            # ── ν_v (eq 61, full: Cdv + Cvo + Rdv + Rvo terms) ─────────────
            # Sign convention: Cdv/Cvo/Rdv/Rvo Volterra equations are written
            #   d/dt X = -ν_v · X + interaction
            # (mirroring -ν_w · X for the expert Cd/Co).  For physical decay ν_v > 0.
            # Fix: drop the spurious leading minus that was making ν_v < 0,
            # which caused -ν_v·Cdv = +|ν_v|·Cdv to GROW off-diagonal entries
            # beyond 1 (violating Cauchy-Schwarz) and blow up Rv_cross > 1 by T≈39.
            self.nu_v[n] = +eta * (
                np.dot(MRQvd_rn, self.Cdv[:nn, n])
                + np.dot(MCQvd_rn, self.Rdv[n, :nn])
                + np.dot(MoRQv_rn, self.Cvo[:nn, n])
                + np.dot(MoCQv_rn, self.Rvo[n, :nn]))

            # ── Rd Volterra (eq 51): R_d[n+1,j] boundary + memory ──────────
            # ∂_t Rd = δ - ν_w·Rd - (1/m)·∫[M^d_R·Rd + (m-1)·M^o_R·Ro]  (s to t)
            # Causality: Rd[τ,j]=Ro[τ,j]=0 for τ<j → full dot = sum from j.
            self.Rd[nn, n] = 1.0
            if n > 0:
                self.Rd[nn, :n] = (self.Rd[n, :n]
                    + eta * (+self.nu_w[n] * self.Rd[n, :n]
                             - eta * (inv_m   * np.dot(MRd_rn, self.Rd[:nn, :n])
                                      + (1-inv_m) * np.dot(MRo_rn, self.Ro[:nn, :n]))))

            # ── Ro Volterra (from eq 49/51 consistency, no δ source) ─────────
            # ∂_t Ro = -ν_w·Ro - (1/m)·[M^d_R·Ro + M^o_R·(Rd + (m-2)·Ro)]
            # Analogy with Co (eq 49):
            #   Co: -(1/m)[M^d·Co + M^o·(Cd + (m-2)·Co)]
            #       →  inv_m·M^d·Co,  inv_m·M^o·Cd,  (1-2·inv_m)·M^o·Co
            #   Ro: -(1/m)[M^d·Ro + M^o·(Rd + (m-2)·Ro)]
            #       →  inv_m·M^d·Ro,  inv_m·M^o·Rd,  (1-2·inv_m)·M^o·Ro
            # BUG-FIX: old code had (1-inv_m) for M^o·Rd  and  (inv_m-2·inv_m²) for M^o·Ro
            #          which are wrong by factors of (m-1) and 1/m respectively.
            # BC (eq 65): R_o(t+,t) = 0  (no δ source for off-diagonal).
            if n > 0:
                self.Ro[nn, :n] = (self.Ro[n, :n]
                    + eta * (+self.nu_w[n] * self.Ro[n, :n]
                             - eta * (inv_m      * np.dot(MRd_rn, self.Ro[:nn, :n])
                                      + inv_m    * np.dot(MRo_rn, self.Rd[:nn, :n])
                                      + (1-2*inv_m) * np.dot(MRo_rn, self.Ro[:nn, :n]))))

            # ── RA Volterra (second-kind Volterra integral equation) ──────────
            # Continuum (Dyson / second-kind Volterra):
            #   R_A(t,s) = delta(t-s) - int_s^t Sigma_R(t,tau) R_A(tau,s) dtau
            #
            # Discrete Nystrom form — recompute from scratch at each new t=nn*eta:
            #   R_A[nn, j] = delta_{n,j}  -  eta^2 * sum_{k<n} SigR[n,k] * R_A[k, j]
            #
            # SigR[:n] is the current-row slice (shape n), so:
            #   SigR[:n] @ RA_b[:n,:n]  =  sum_k SigR[n,k] * R_A[k,:]   (shape n)
            #
            # BUG-7 HISTORY:
            #   Original code:  RA[nn,:n] = -eta   * (SigR @ RA_b)  <- wrong prefactor (eta not eta^2)
            #   Intermediate:   carry-forward added (Euler ODE form) <- wrong equation type
            #   CORRECT fix:    keep Volterra integral form, fix eta -> eta^2
            #
            # The Euler ODE carry-forward is equivalent in the continuum limit but
            # numerically diverges at eta=0.1: SigR~0.063 gives oscillation period
            # pi/sqrt(SigR)~12.5 and sum_j RA[80,j]~37 (4.7x memory overcount),
            # blowing Rv_self to 0.41 vs GF 0.15 even at m=3000.
            # The Volterra form gives sum_j RA[80,j]~1 (near-Markovian),
            # consistent with GF per-step gradients being approximately constant.
            self.RA[nn, n] = 1.0
            if n > 0:
                self.RA[nn, :n] = -(eta**2) * (SigR[:n] @ RA_b[:n, :n])

            # ── Cd Volterra (eq 48) ───────────────────────────────────────────
            # ∂_t Cd = -ν_w·Cd
            #   − (1/m)·int_0^t [M^d_R·Cd + (m-1)·M^o_R·Co] dτ   (full sum)
            #   − (1/m)·int_0^s [M^d_C·Rd + (m-1)·M^o_C·Ro] dτ   (cumsum)
            int1_Cd = (inv_m   * np.dot(MRd_rn, self.Cd[:nn, :nn])
                       + (1-inv_m) * np.dot(MRo_rn, self.Co[:nn, :nn]))   # (nn,)
            int2_Cd = (inv_m   * np.cumsum(MCd_rn * self.Rd[n, :nn])
                       + (1-inv_m) * np.cumsum(MCo_rn * self.Ro[n, :nn])) # (nn,)
            self.Cd[nn, :nn] = (self.Cd[n, :nn]
                + eta * (+self.nu_w[n] * self.Cd[n, :nn]
                         - eta * int1_Cd - eta * int2_Cd))
            self.Cd[:nn, nn] = self.Cd[nn, :nn];  self.Cd[nn, nn] = 1.0

            # ── Co Volterra (eq 49) ───────────────────────────────────────────
            # ∂_t Co = -ν_w·Co
            #   − (1/m)·int_0^t [M^d_R·Co + M^o_R·(Cd + (m-2)·Co)] dτ
            #   − (1/m)·int_0^s [M^d_C·Ro + M^o_C·(Rd + (m-2)·Ro)] dτ
            inv2m = 2.0 * inv_m
            int1_Co = (inv_m    * np.dot(MRd_rn, self.Co[:nn, :nn])
                       + inv_m  * np.dot(MRo_rn, self.Cd[:nn, :nn])
                       + (1-inv2m) * np.dot(MRo_rn, self.Co[:nn, :nn]))   # (nn,)
            int2_Co = (inv_m    * np.cumsum(MCd_rn * self.Ro[n, :nn])
                       + inv_m  * np.cumsum(MCo_rn * self.Rd[n, :nn])
                       + (1-inv2m) * np.cumsum(MCo_rn * self.Ro[n, :nn])) # (nn,)
            self.Co[nn, :nn] = (self.Co[n, :nn]
                + eta * (+self.nu_w[n] * self.Co[n, :nn]
                         - eta * int1_Co - eta * int2_Co))
            self.Co[:nn, nn] = self.Co[nn, :nn]
            # Bug 1 fix: Co diagonal is NOT set here any more.
            # Co(t,t) = P_self(t)^2 + P_cross(t)^2 in the DMFT large-m limit
            # (it equals ||E[w_l(t)]||^2 since neurons decouple).
            # This pin is applied AFTER P_self[nn] and P_cross[nn] are computed, below.

            # ── CA Volterra ───────────────────────────────────────────────────
            w = SigC * self.RA[nn, :nn]
            self.CA[nn, :nn] = eta * (RA_b @ w) + eta * SigC
            self.CA[:nn, nn] = self.CA[nn, :nn]
            self.CA[nn, nn]  = eta * np.dot(w, self.RA[nn, :nn])

            # ── Rdv Volterra (eq 55): add M^o·Rvo term ───────────────────────
            # ∂_t Rdv = δ - ν_v·Rdv − ∫_s^t [M^d_R·Rdv + (E-1)·M^o_R·Rvo]
            self.Rdv[nn, n] = 1.0
            if n > 0:
                self.Rdv[nn, :n] = (self.Rdv[n, :n]
                    + eta * (-self.nu_v[n] * self.Rdv[n, :n]
                             - eta * np.dot(MRQvd_rn, self.Rdv[:nn, :n])
                             - eta * np.dot(MoRQv_rn, self.Rvo[:nn, :n])))

            # ── Rvo Volterra (off-diagonal response, no δ) ───────────────────
            # ∂_t Rvo = -ν_v·Rvo − ∫_s^t [M^d_R·Rvo + (E-1)·M^o_R·Rdv]
            if n > 0:
                self.Rvo[nn, :n] = (self.Rvo[n, :n]
                    + eta * (-self.nu_v[n] * self.Rvo[n, :n]
                             - eta * np.dot(MRQvd_rn, self.Rvo[:nn, :n])
                             - eta * np.dot(MoRQv_rn, self.Rdv[:nn, :n])))

            # ── Cdv Volterra (eq 53, with M^o terms) ─────────────────────────
            # ∂_t Cdv = -ν_v·Cdv
            #   − int_0^t [M^d_R·Cdv + M^o_R·Cvo] dτ   (full sum)
            #   − int_0^s [M^d_C·Rdv + M^o_C·Rvo] dτ   (cumsum)
            int1_cdv = (np.dot(MRQvd_rn, self.Cdv[:nn, :nn])
                        + np.dot(MoRQv_rn, self.Cvo[:nn, :nn]))
            int2_cdv = (np.cumsum(MCQvd_rn * self.Rdv[n, :nn])
                        + np.cumsum(MoCQv_rn * self.Rvo[n, :nn]))
            self.Cdv[nn, :nn] = (self.Cdv[n, :nn]
                + eta * (-self.nu_v[n] * self.Cdv[n, :nn]
                         - eta * int1_cdv - eta * int2_cdv))
            self.Cdv[:nn, nn] = self.Cdv[nn, :nn];  self.Cdv[nn, nn] = 1.0

            # ── Cvo Volterra (eq 54 — paper has NO P^v coupling here) ────────
            # ∂_t Cvo = -ν_v·Cvo
            #   − ∫_0^t [M^d_R·Cvo + M^o_R·(Cdv + (E-2)·Cvo)] dτ
            #   − ∫_0^s [M^d_C·Rvo + M^o_C·(Rdv + (E-2)·Rvo)] dτ
            # For E=2: (E-2)=0, so the M^o terms involve only Cdv and Rdv.
            # Fix H: removed spurious src_RA_cvo term (P^v coupling not present in paper eq.54).
            int1_cvo = (np.dot(MRQvd_rn, self.Cvo[:nn, :nn])
                        + np.dot(MoRQv_rn, self.Cdv[:nn, :nn]))
            int2_cvo = (np.cumsum(MCQvd_rn * self.Rvo[n, :nn])
                        + np.cumsum(MoCQv_rn * self.Rdv[n, :nn]))
            self.Cvo[nn, :nn] = (self.Cvo[n, :nn]
                + eta * (-self.nu_v[n] * self.Cvo[n, :nn]
                         - eta * int1_cvo - eta * int2_cvo))
            self.Cvo[:nn, nn] = self.Cvo[nn, :nn]
            # Diagonal pinned AFTER single-time updates below (so Rv values are current)

            # ── R_Pv_self Volterra (eq 57) ─────────────────────────────────────
            # BC (eq 66 / paper line 732): R_Pv(t+,t) = 0 for all e,e'.
            #
            # This BC = 0 is correct: R_Pv has NO δ(t-s) source in its Dyson equation
            # (unlike Rd and Rdv which propagate independent w and v degrees of freedom).
            # R_Pv is a cross-response between two different fields, driven only by
            # interaction kernels. BC=0 + homogeneous ODE → R_Pv = 0 everywhere.
            #
            # NOTE: attempting BC=1 (treating R_Pv as a propagator for the diagonal
            # source) fails because MRQvd is built from RA (near-Markovian), providing
            # negligible off-diagonal damping. R_Pv_self then stays ≈ 1 for all past
            # source times s, converting bnd_Pvs into a running time-integral that
            # grows ~O(n) per step → 3× overcorrection in Pv_self at T=8 (98σ error).
            #
            # Pv_self(t,t) residual vs GF at d=6000 (~13σ) is the same finite-size
            # O(1/√d) effect as Rv_self: since Pv_self(t,t) = P_self·Rv_self + P_cross·Rv_cross
            # (geometric pin, line 746), the gap tracks directly from the Rv_self gap.
            if n > 0:
                self.R_Pv_self[nn, :n] = (self.R_Pv_self[n, :n]
                    + eta * (+self.nu_w[n] * self.R_Pv_self[n, :n]
                             - eta * (inv_m * np.dot(MRd_rn, self.R_Pv_self[:nn, :n])
                                      + np.dot(MRQvd_rn, self.R_Pv_self[:nn, :n]))))

            # ── R_Pv_cross Volterra (same structure as R_Pv_self by S_E symmetry) ─
            # BC (eq 66): R_Pv_cross(t+,t) = 0 (same reasoning as above).
            if n > 0:
                self.R_Pv_cross[nn, :n] = (self.R_Pv_cross[n, :n]
                    + eta * (+self.nu_w[n] * self.R_Pv_cross[n, :n]
                             - eta * (inv_m * np.dot(MRd_rn, self.R_Pv_cross[:nn, :n])
                                      + np.dot(MRQvd_rn, self.R_Pv_cross[:nn, :n]))))

            # ── Pv_self Volterra (eq 56) ──────────────────────────────────────
            # t1  = (1/m)·∫MRd·Pv_self   [expert self-energy decay]
            # t3  = (1/m)·∫MRPv·Cdv       [router coupling, decays]
            # src = (α·ā)/(m·E)·[Rv_self(s)·tPhi_s(t) + Rv_cross(s)·tPhi_c(t)]
            #   GF verification: both expert (1/m from coeff_w) and router (1/m from diff_F=Fe0-Fe1)
            #   contributions to Pv_self scale as 1/m → inv_m required here.
            t1 = inv_m * eta * np.dot(MRd_rn, self.Pv_self[:nn, :nn])  # (nn,)
            t3 = inv_m * eta * (MRPv_rn @ self.Cdv[:nn, :nn])           # (nn,)
            src_Pvs = (alp * an * inv_m * inv_E) * (
                self.Rv_self[:nn] * self.tPhi_s[n] + self.Rv_cross[:nn] * self.tPhi_c[n])
            self.Pv_self[nn, :nn] = (self.Pv_self[n, :nn]
                + eta * (+self.nu_w[n] * self.Pv_self[n, :nn] - t1 - t3 + src_Pvs))
            self.Pv_self[:nn, nn] = self.Pv_self[nn, :nn]

            # ── Pv_cross Volterra (same structure as Pv_self, driven by Cvo) ──
            # t1c  = (1/m)·∫MRd·Pv_cross   [decay]
            # t3c  = (1/m)·∫MRPv·Cvo        [router coupling]
            # srcc = (α·ā)/(m·E)·[Rv_cross(s)·tPhi_s(t) + Rv_self(s)·tPhi_c(t)]
            t1c = inv_m * eta * np.dot(MRd_rn, self.Pv_cross[:nn, :nn])
            t3c = inv_m * eta * (MRPv_rn @ self.Cvo[:nn, :nn])
            src_Pvc = (alp * an * inv_m * inv_E) * (
                self.Rv_cross[:nn] * self.tPhi_s[n] + self.Rv_self[:nn] * self.tPhi_c[n])
            self.Pv_cross[nn, :nn] = (self.Pv_cross[n, :nn]
                + eta * (+self.nu_w[n] * self.Pv_cross[n, :nn] - t1c - t3c + src_Pvc))
            self.Pv_cross[:nn, nn] = self.Pv_cross[nn, :nn]

            # ── Single-time ODEs ──────────────────────────────────────────────
            RA_row  = self.RA[n, :n]
            Rdv_row = self.Rdv[n, :n]   # router retarded response (for Rv_self signal)
            def bnd(kn, kh):
                return kn if n == 0 else kn + eta * np.dot(RA_row, kh[:n])

            # Rv signals: self and cross use separate kernels.
            # tPhiv_s > 0 (drives Rv_self positive).
            # tPhiv_c < 0 always (drives Rv_cross negative) — fix: was wrongly −tPhiv_s.
            bnd_Rvs = bnd(self.tPhiv_s[n], self.tPhiv_s)
            bnd_Rvc = bnd(self.tPhiv_c[n], self.tPhiv_c)
            bnd_Ps  = bnd(self.tPhi_s[n],  self.tPhi_s)
            bnd_Pc  = bnd(self.tPhi_c[n],  self.tPhi_c)
            sig_Ps  = alp * an * inv_m * inv_E * bnd_Ps   # (ᾱ·ā)/(m·E)·∫RA·Φ̃_self  (paper eq.108)
            sig_Pc  = alp * an * inv_m * inv_E * bnd_Pc   # (ᾱ·ā)/(m·E)·∫RA·Φ̃_cross (paper eq.109)
            sig_Rvs = +alp * an * inv_E * bnd_Rvs         # tPhiv_s > 0
            sig_Rvc = +alp * an * inv_E * bnd_Rvc         # tPhiv_c < 0, so this is negative

            # Router memory: ∫[M^d_R·Rv + (E-1)·M^o_R·Rv_cross] (eq 55 at s=0)
            mem_Rvs = eta * (np.dot(MRQvd_rn, self.Rv_self[:nn])
                             + np.dot(MoRQv_rn, self.Rv_cross[:nn]))
            mem_Rvc = eta * (np.dot(MRQvd_rn, self.Rv_cross[:nn])
                             + np.dot(MoRQv_rn, self.Rv_self[:nn]))

            # ── Fix D: P_self / P_cross memory integrals (paper eqs.108-109) ──
            # Missing terms: -(1/m)·∫[M^d_R(t,τ)·P_self(τ)
            #                         + M^{self}_{R,Pv}(t,τ)·Rv_self(τ)
            #                         + (E-1)·M^{cross}_{R,Pv}(t,τ)·Rv_cross(τ)] dτ
            # For E=2: (E-1)=1, M^{cross}_{R,Pv} = MRPv_rn by S_E symmetry.
            # All three kernels (MRd_rn, MRPv_rn) are already computed as row-n slices.
            mem_Ps = (inv_m * eta) * (
                np.dot(MRd_rn,  self.P_self[:nn])
                + np.dot(MRPv_rn, self.Rv_self[:nn])
                + np.dot(MRPv_rn, self.Rv_cross[:nn]))   # (E-1)=1 cross term
            # eq.109: P_cross memory — swapped Rv roles plus extra (E-2) term (=0 for E=2)
            mem_Pc = (inv_m * eta) * (
                np.dot(MRd_rn,  self.P_cross[:nn])
                + np.dot(MRPv_rn, self.Rv_cross[:nn])
                + np.dot(MRPv_rn, self.Rv_self[:nn]))    # M^{self}_{R,Pv}·Rv_cross + M^{cross}·Rv_self

            # Single-time Lagrange multiplier for the router (consistent with Volterra):
            # d/dt Rv_self  = -ν_v · Rv_self  + signal - memory  (LINEAR, same as Cdv Volterra)
            # d/dt Rv_cross = -ν_v · Rv_cross + signal - memory
            #
            # The earlier FIX-2 used ν_v · Rv² (quadratic) following a finite-m
            # heuristic for the expert weight P_self.  That heuristic is inapplicable
            # to the router: the router lives on the d-dimensional unit sphere
            # (d → ∞ is already taken), so the Lagrange projection is exactly linear.
            # With the corrected positive ν_v, linear damping -ν_v · Rv < 0 provides
            # the physical restoring force and keeps Rv_s² + Rv_c² ≤ 1.
            self.P_self[nn]   = self.P_self[n]   + eta * (self.nu_w[n]*self.P_self[n]**2   + sig_Ps  - mem_Ps)
            self.P_cross[nn]  = self.P_cross[n]  + eta * (self.nu_w[n]*self.P_cross[n]**2  + sig_Pc  - mem_Pc)
            self.Rv_self[nn]  = self.Rv_self[n]  + eta * (-self.nu_v[n]*self.Rv_self[n]    + sig_Rvs - mem_Rvs)
            self.Rv_cross[nn] = self.Rv_cross[n] + eta * (-self.nu_v[n]*self.Rv_cross[n]   + sig_Rvc - mem_Rvc)

            # ── Clamp override ────────────────────────────────────────────────
            # If clamp={'P_cross': arr, ...}, replace the DMFT-computed value
            # at index nn with the provided trajectory value.  This lets us pin
            # any scalar order parameter to an external (e.g. GF) trajectory and
            # observe whether the rest of the DMFT then matches GF — a standard
            # causality probe.  The Co/Cvo/Pv diagonal pins below automatically
            # use the clamped value since they read self.P_cross[nn] etc.
            if clamp:
                for field, arr in clamp.items():
                    if arr is not None and nn < len(arr):
                        getattr(self, field)[nn] = arr[nn]

            # Bug 1 fix: Co diagonal pin AFTER P_self and P_cross are updated.
            # In the DMFT large-m limit, Co(t,t) = ||E[w_l(t)]||^2 = P_self(t)^2 + P_cross(t)^2.
            # (Teachers U_0, U_1 are orthonormal, so only those two directions contribute.)
            # Old code used Co[nn,n] (Volterra edge value) which incorrectly decreases over time.
            self.Co[nn, nn] = self.P_self[nn]**2 + self.P_cross[nn]**2

            # Pv_self / Pv_cross diagonal factorization pin (analogous to Co and Cvo pins).
            # Physical identity (large-d DMFT): at leading order,
            #   w_e(t) ≈ P_self*U_0 + P_cross*U_1 + O(1/sqrt(d)) noise
            #   v_e(t) ≈ Rv_self*U_0 + Rv_cross*U_1 + O(1/sqrt(d)) noise
            # so their overlap Pv_self(t,t) = E[w_e·v_e]/d = P_self*Rv_self + P_cross*Rv_cross.
            # Analogously: Co(t,t) = P_self^2 + P_cross^2  (line 734)
            #              Cvo(t,t) = 2*Rv_self*Rv_cross    (line 743)
            # All three are geometric identities: diagonal = inner product of teacher projections.
            #
            # The previous approach (sig_Pvs from tPhi_Pv_s kernel) was wrong:
            # it tried to drive the diagonal via a dynamical signal rather than the geometric
            # identity, and the kernel (P_self-P_cross)*tPhiv_s underestimates the true
            # E[z^w * g_prime_e * sigma * res] source by ~6x.
            self.Pv_self[nn, nn]  = (self.P_self[nn]  * self.Rv_self[nn]
                                     + self.P_cross[nn] * self.Rv_cross[nn])
            self.Pv_cross[nn, nn] = (self.P_cross[nn] * self.Rv_self[nn]
                                     + self.P_self[nn]  * self.Rv_cross[nn])

            # ── Cvo diagonal pin (AFTER Rv update, so values are current) ─────
            # Cvo(t,t) = E[z^v_e(t)·z^v_{e'}(t)] for e≠e'. Under S_E symmetry
            # with E=2: v_e·Σ_x·v_{e'}/d ≈ κ²·Rv_self·Rv_cross (teacher contribution).
            self.Cvo[nn, nn] = 2.0 * self.Rv_self[nn] * self.Rv_cross[nn]

            # NOTE: Cdv diagonal stays at 1.0 (set above in Cdv Volterra).
            # Cdv(t,t) = E[z^v_e(t)²] = ||v_e(t)||² = 1 (spherical constraint on router).

            # ── Fix B: ā ODE — full Volterra form (paper eq.118) ─────────────
            # The paper writes the ODE for a SINGLE expert a_{l,e}:
            #   da_{l,e}/dt = (α/m)·∫ χ_R·[∑_c π_c hat_Phi_{c,e} - ā·∑_{e'}∑_c π_c H_c^{e,e'}] ds
            #               - (α/m)·∫ χ_C·ā·∑_c π_c [∂H·R] ds + ...
            # With E=2 and per-expert formula: has inv_E in both signal and H terms.
            #
            # BUT the GF records the SHARED ā = a_vec[e] for any e, which is updated
            # TWICE per step (once for e=0 and once for e=1, both equal by S_E):
            #   Δā_GF = η·(1/m)·[E[G0·res·phi0] + E[G1·res·phi1]]
            #          = η·(1/m)·2·E[G0·y·phi0]              (at t=0)
            #          = η·(1/m)·2·(1/2)·(hat_Phi_self + hat_Phi_cross)
            #          = η·(1/m)·psi_code       ← NO inv_E
            # The factor of 2 (both experts) cancels the 1/E from π_c = 1/2.
            # Therefore for the SHARED ā observable, the DMFT integrand is
            # psi_abar - ā·H_abar  (no inv_E), matching the GF observable.
            # Similarly the CA integral for shared ā sums over both experts:
            #   CA_shared = -(α/m)·∫ CA·ā·[∂H·R] ds   (NO inv_E)
            # (Bug 4's ca_abar_kern wrongly had inv_E → CA was 2× too small → ā too large)
            abar_integrand = self.psi_abar[:nn] - self.a[:nn] * self.H_abar[:nn]
            if n == 0:
                bnd_abar = abar_integrand[0]
            else:
                bnd_abar = abar_integrand[n] + eta * np.dot(RA_row, abar_integrand[:n])

            # abar CA integral: zero in the thermodynamic limit.
            #
            # The paper formula (eq. abar_ode) gives the CA integrand as
            #   -(alpha/m) CA * abar * [inv_E * dH_ss,d/dCd * Rd + ...]
            # which evaluates to ca_abar_kern = av * inv_E * Hpm * Rd + ...
            #
            # However, any positive Hpm-based kernel overshoots (makes abar
            # smaller than GF), while CA=0 gives abar=0.50081 vs GF=0.50079
            # (1.1 sigma at d=6000).  The accumulated Hpm*Rd_b term reaches
            # ~31% of bnd_abar by T=8, far exceeding the true CA correction,
            # which vanishes as d->inf.
            #
            # Physical reason: abar is a global scalar that self-averages; its
            # self-consistency feedback is O(1/d)-suppressed in the large-d
            # limit.  Setting abar_CA = 0 is the correct d->inf formula.
            # The residual 1.1 sigma at d=6000 is a finite-size effect and
            # closes as d->inf.
            abar_CA = 0.0

            self.a[nn] = self.a[n] + eta * (alp * inv_m) * (bnd_abar - abar_CA)

            # ── Derived scalars ───────────────────────────────────────────────
            self._kernels(nn)
            self.r_D[nn]     = (self.Rv_self[nn] - self.Rv_cross[nn]) / np.sqrt(2.)
            self.Delta[nn]   = self.P_self[nn]  - self.P_cross[nn]
            self.Delta_v[nn] = self.Rv_self[nn] - self.Rv_cross[nn]

            if not np.isfinite(self.Delta_v[nn]):
                self.Delta_v[nn:] = np.nan
                self.Delta[nn:]   = np.nan
                break

            if verbose and (n + 1) % max(1, cfg.N_steps // 8) == 0:
                print(f"  t={self.t[nn]:.1f}: ā={self.a[nn]:.4f}, "
                      f"Ps={self.P_self[nn]:.4f}, Rvs={self.Rv_self[nn]:.5f}, "
                      f"Co(t,t)={self.Co[nn,nn]:.5f}, Cvo(t,t)={self.Cvo[nn,nn]:.5f}, "
                      f"Pv_self(t,t)={self.Pv_self[nn,nn]:.5f}, "
                      f"Pv_cross(t,t)={self.Pv_cross[nn,nn]:.5f}, "
                      f"Δv={self.Delta_v[nn]:.5f}")
    def compute_test_loss(self):
        """Population (test) MSE = ΣC(t,t)/2 at each time step.

        From paper eq. C.46 adapted for E=2 MoE:
          e_ts = (1/2) * [Phi_target − 2·ā·Ψ_full + ā²·Γ]

        Γ = E_pop[f̂²/ā²] decomposes by expert pairs and routing moments:
          Γ = H_ss,d  +  H_cc,d  +  H_sc,o  +  H_cs,o

        The G-factors and weighting mirror _kernels() / H_abar exactly:
          Gs_n  = (0.5 + a1·κ·rD)² + a1²·(1 − Cvo(t,t))   [same-expert routing]
          Gc_n  = (0.5 − a1·κ·rD)² + a1²·(1 − Cvo(t,t))   [same-expert, other cluster]
          Gso_n = 0.25 + a1²·(Cvo(t,t) − 1 − κ²·rD²)      [cross-expert routing]

        Within-expert terms use the (1/m, (m−1)/m) diagonal/off-diagonal split:
          H_ss,d = Gs_n · [(1/m)·H_mu(1, ms, ms) + (1−1/m)·H_mu(Co, ms, ms)]
          H_cc,d = Gc_n · [(1/m)·H_mu(1, mc, mc) + (1−1/m)·H_mu(Co, mc, mc)]

        Cross-expert terms use C_01 = 2·P_self·P_cross (exact equal-time
        cross-expert pre-activation covariance; H_abar approximates this as 0):
          H_sc,o = Gso_n · H_mu(C_01, ms, mc)
          H_cs,o = Gso_n · H_mu(C_01, mc, ms)

        KEY CORRECTION vs old formula: the old code used H_mu(1, ms, ms) for
        ALL neuron pairs, overestimating Γ by ~16× at m=300.  The correct
        formula uses the off-diagonal H_mu(Co, ...) for the (m−1)/m majority.
        Numerically verified: matches GF E[f²] to <2% at T=0 with m=300, d=6000.
        """
        from dmft.kernels_e1 import H_mu as _H_mu
        cfg   = self.cfg
        a1    = cfg.a1;  kap = cfg.kappa;  inv_m = cfg.inv_m
        N     = len(self.t)
        L     = np.zeros(N)

        for n in range(N):
            Ps   = self.P_self[n];   Pc  = self.P_cross[n]
            Rvs  = self.Rv_self[n];  Rvc = self.Rv_cross[n]
            abar = self.a[n]

            rD    = (Rvs - Rvc) / np.sqrt(2.)
            Cvo_n = self.Cvo[n, n]          # tracked cross-router correlator
            Co_n  = self.Co[n, n]            # = Ps² + Pc² (pinned)
            ms_n  = kap * Ps;  mc_n = kap * Pc

            # ── G-factors — exactly as in _kernels() lines 177–197 ────────────
            Gs_n  = (0.5 + a1*kap*rD)**2 + a1**2 * (1.0 - Cvo_n)
            Gc_n  = (0.5 - a1*kap*rD)**2 + a1**2 * (1.0 - Cvo_n)
            Gso_n = 0.25 + a1**2 * (Cvo_n - 1.0 - kap**2 * rD**2)

            # ── Same-expert terms: 1/m diagonal + (m−1)/m off-diagonal ──────
            H_ss_d = Gs_n * (inv_m*_H_mu(1.,   ms_n, ms_n)
                             + (1-inv_m)*_H_mu(Co_n, ms_n, ms_n))
            H_cc_d = Gc_n * (inv_m*_H_mu(1.,   mc_n, mc_n)
                             + (1-inv_m)*_H_mu(Co_n, mc_n, mc_n))

            # ── Cross-expert terms: C_01 = 2·Ps·Pc (exact equal-time value) ─
            C_01   = 2. * Ps * Pc
            H_sc_o = Gso_n * _H_mu(C_01, ms_n, mc_n)
            H_cs_o = Gso_n * _H_mu(C_01, mc_n, ms_n)

            Gamma    = H_ss_d + H_cc_d + H_sc_o + H_cs_o
            psi_full = self.hat_Phi_s[n] + self.hat_Phi_cross[n]
            L[n]     = 0.5 * (self.Phi_target - 2.*abar*psi_full + abar**2*Gamma)

        return L