"""
moe_sim/dmft/kernels_e1.py
==========================
Closed-form DMFT kernels for E=1 (single expert, no routing).

Activation: sigma = phi = erf(z/sqrt(2))

From §13 (E=2, quad-probit section), the key formula is:
  h(q) = (2/pi) * arcsin(q/2)

This is derived from the bivariate orthant probability for erf (NOT arcsin(q)),
and was confirmed in the document at line ~1443.

Key values:
  h(0)  = 0
  h(1)  = 1/3
  h'(0) = 1/pi
  h'(1) = 2/(pi*sqrt(3)) ≈ 0.3676

Hermite coefficients (§13.1):
  hat_sigma_1 = hat_phi_1 = 1/sqrt(pi)
  hat_sigma_1^2 = 1/pi = h'(0)  [consistency check ✓]

Information exponent k* = 1  =>  specialization starts at O(1) time.
"""

import numpy as np


# ── Core kernel ──────────────────────────────────────────────────────────────

def h(q):
    """
    Student-student (and student-teacher) kernel for erf * erf.

    h(q) = (2/pi) * arcsin(q/2)

    Verified: h(0)=0, h(1)=1/3, h'(0)=1/pi, derived from bivariate orthant
    probability formula for the product of two erfs.
    """
    q = np.asarray(q, dtype=float)
    # Clip slightly inside [-1, 1] to avoid NaN at exact boundaries
    q_safe = np.clip(q, -1.0 + 1e-12, 1.0 - 1e-12)
    return (2.0 / np.pi) * np.arcsin(q_safe / 2.0)


def h_prime(q):
    """
    First derivative: h'(q) = 1 / (pi * sqrt(1 - q^2/4))

    Satisfies: h'(0) = 1/pi = hat_sigma_1^2  (Hermite k=1 coefficient squared).
    """
    q = np.asarray(q, dtype=float)
    q_safe = np.clip(q, -2.0 + 1e-12, 2.0 - 1e-12)
    return 1.0 / (np.pi * np.sqrt(1.0 - q_safe**2 / 4.0))


def h_double_prime(q):
    """
    Second derivative: h''(q) = q / (2*pi*(1 - q^2/4)^(3/2))
    """
    q = np.asarray(q, dtype=float)
    q_safe = np.clip(q, -2.0 + 1e-12, 2.0 - 1e-12)
    return q_safe / (2.0 * np.pi * (1.0 - q_safe**2 / 4.0)**1.5)


# ── Derived quantities ────────────────────────────────────────────────────────

# These are defined for notational clarity; they're just h evaluated at different args.

def hat_Phi(P):
    """Student-teacher kernel: hat_Phi(P) = h(P) = (2/pi)*arcsin(P/2)."""
    return h(P)


def tilde_Phi(P):
    """Derivative kernel: tilde_Phi(P) = d/dP hat_Phi(P) = h'(P)."""
    return h_prime(P)


# Constants
PHI_TARGET = h(1.0)   # = 1/3; teacher self-overlap: E[phi^2(lambda)] with lambda~N(0,1)
H_AT_ONE   = h(1.0)   # = 1/3; student self-overlap at C_d(t,t)=1
HP_AT_ONE  = h_prime(1.0)  # = 2/(pi*sqrt(3)) ≈ 0.3676


# ── Self-energies at E=1 ─────────────────────────────────────────────────────

def sigma_C(a_t, a_s, P_t, P_s, Cd_ts):
    """
    Self-energy Sigma_C(t,s) at E=1, tau=0, no cluster means.

    Sigma_C(t,s) = Phi_target - a(t)*hat_Phi(t) - a(s)*hat_Phi(s) + a(t)*a(s)*h(C_d(t,s))
    """
    return (PHI_TARGET
            - a_t * h(P_t)
            - a_s * h(P_s)
            + a_t * a_s * h(Cd_ts))


def sigma_R(a_t, a_s, Cd_ts, Rd_ts):
    """
    Self-energy Sigma_R(t,s) at E=1, large-m.

    Sigma_R(t,s) = a(t)*a(s)*h'(C_d(t,s))*R_d(t,s)
    """
    return a_t * a_s * h_prime(Cd_ts) * Rd_ts


# ── Test error formula ────────────────────────────────────────────────────────

def test_error_e1(a_t, P_t):
    """
    Test error at E=1:
    e_ts(t) = (1/2) * [Phi_target - 2*a(t)*hat_Phi(t) + a(t)^2 * h(1)]
            = (1/2) * [1/3 - 2*a(t)*h(P(t)) + a(t)^2/3]
    """
    return 0.5 * (PHI_TARGET - 2.0 * a_t * h(P_t) + a_t**2 * H_AT_ONE)


# ── Verification ─────────────────────────────────────────────────────────────

def verify_kernel():
    """Quick sanity checks on the kernel formulas."""
    print("Kernel verification:")
    print(f"  h(0)   = {h(0):.6f}  (expected 0)")
    print(f"  h(1)   = {h(1):.6f}  (expected {1/3:.6f})")
    print(f"  h'(0)  = {h_prime(0):.6f}  (expected {1/np.pi:.6f} = 1/pi)")
    print(f"  h'(1)  = {h_prime(1):.6f}  (expected {2/(np.pi*np.sqrt(3)):.6f})")
    print(f"  hat_sigma_1^2 = {1/np.pi:.6f}  =? h'(0) = {h_prime(0):.6f}  {'✓' if abs(h_prime(0) - 1/np.pi) < 1e-10 else '✗'}")
    print(f"  Phi_target = {PHI_TARGET:.6f}  (= h(1) = 1/3)")
    # Hermite expansion check: h(q) ≈ (1/pi)*q + ... for small q
    q_small = 0.01
    h_approx = (1/np.pi)*q_small
    print(f"  h({q_small}) = {h(q_small):.8f},  (1/pi)*q = {h_approx:.8f},  diff = {h(q_small)-h_approx:.2e}")


if __name__ == "__main__":
    verify_kernel()

# ── Cluster-mean (shifted) kernels ──────────────────────────────────────────

from numpy.polynomial.hermite import hermgauss as _hermgauss
from scipy.special import erf as _erf_sc; erf = _erf_sc

_N_GAUSS = 64
_ZZ, _WW = _hermgauss(_N_GAUSS)
_ZZ *= np.sqrt(2.0); _WW /= np.sqrt(np.pi)  # nodes/weights for N(0,1)


def H_mu(q, m1, m2):
    """E[phi(m1+G1)*phi(m2+G2)], (G1,G2)~BivN(0,[[1,q],[q,1]]).
    Reduces to h(q) when m1=m2=0.
    """
    q = float(np.clip(q, -1+1e-10, 1-1e-10))
    phi2_cond = erf((m2 + q*_ZZ) / np.sqrt(2.0*(2.0 - q*q)))
    return float(np.dot(_WW, erf((m1+_ZZ)/np.sqrt(2.0)) * phi2_cond))


def dH_mu_dm1(q, m1, m2):
    """E[phi'(m1+G1)*phi(m2+G2)]  (partial deriv of H_mu w.r.t. m1)."""
    q = float(np.clip(q, -1+1e-10, 1-1e-10))
    phi2_cond = erf((m2 + q*_ZZ) / np.sqrt(2.0*(2.0 - q*q)))
    phip1 = np.sqrt(2.0/np.pi) * np.exp(-0.5*(m1+_ZZ)**2)
    return float(np.dot(_WW, phip1 * phi2_cond))


def dH_mu_dm2(q, m1, m2):
    """E[phi(m1+G1)*phi'(m2+G2)]  (partial deriv of H_mu w.r.t. m2)."""
    q = float(np.clip(q, -1+1e-10, 1-1e-10))
    denom = np.sqrt(2.0 - q*q)
    phip2_cond = np.sqrt(2.0/np.pi) * np.exp(-(m2+q*_ZZ)**2 / (2.0*(2.0-q*q))) / denom
    return float(np.dot(_WW, erf((m1+_ZZ)/np.sqrt(2.0)) * phip2_cond))


def H_mu_prime(q, m1, m2):
    """E[phi'(m1+G1)*phi'(m2+G2)] = d H_mu / dq  (by Price's theorem)."""
    q = float(np.clip(q, -1+1e-10, 1-1e-10))
    denom = np.sqrt(2.0 - q*q)
    phip1 = np.sqrt(2.0/np.pi) * np.exp(-0.5*(m1+_ZZ)**2)
    phip2_cond = np.sqrt(2.0/np.pi) * np.exp(-(m2+q*_ZZ)**2 / (2.0*(2.0-q*q))) / denom
    return float(np.dot(_WW, phip1 * phip2_cond))


def Phi_target_mu(kappa):
    """E[phi(kappa+G)^2] for G~N(0,1)  — teacher variance with cluster mean."""
    return float(np.dot(_WW, erf((kappa+_ZZ)/np.sqrt(2.0))**2))

# ── Lambda-weighted kernels for tPhiv and tPhi ────────────────────────────────
# These include the factor (m2+G2) = lambda_c in the expectation,
# which arises because the gradient projects x onto U_e.
# At kappa=0 these are ZERO by parity — correctly killing the spurious signal.

def _E_y_phi_cond(mu_cond, sig2_cond):
    """E[(mu+G)*phi(mu+G)] for G~N(0, sig2_cond), vectorised over mu_cond."""
    # = mu*E[phi(mu+G)] + sig2*E[phi'(mu+G)]
    # E[phi(mu+G)] = erf(mu/sqrt(2*(1+sig2)))
    # E[phi'(mu+G)] = sqrt(2/pi)*exp(-mu^2/(2*(1+sig2)))/sqrt(1+sig2)
    denom = np.sqrt(2.0*(1.0 + sig2_cond))
    E_phi  = erf(mu_cond / denom)
    E_phip = np.sqrt(2.0/np.pi) * np.exp(-mu_cond**2 / (2.0*(1.0+sig2_cond))) / np.sqrt(1.0+sig2_cond)
    return mu_cond * E_phi + sig2_cond * E_phip


def H_mu_lam(q, m1, m2):
    """E[phi(m1+G1) * phi(m2+G2) * (m2+G2)]
    = the lambda-weighted bivariate kernel for tPhiv.
    Reduces to 0 at m1=m2=0 (parity), unlike H_mu.
    """
    q = float(np.clip(q, -1+1e-10, 1-1e-10))
    mu_cond   = m2 + q*_ZZ          # conditional mean of G2|G1=z
    sig2_cond = 1.0 - q*q
    Ey_phi    = _E_y_phi_cond(mu_cond, sig2_cond)   # E[(m2+G2)*phi(m2+G2)|G1=z]
    phi1      = erf((m1+_ZZ)/np.sqrt(2.0))
    return float(np.dot(_WW, phi1 * Ey_phi))


def dH_mu_dm1_lam(q, m1, m2):
    """E[phi'(m1+G1) * phi(m2+G2) * (m2+G2)]
    = the lambda-weighted dH/dm1 kernel for tPhi (P_self ODE).
    Reduces to 0 at m1=m2=0 (parity).
    """
    q = float(np.clip(q, -1+1e-10, 1-1e-10))
    mu_cond   = m2 + q*_ZZ
    sig2_cond = 1.0 - q*q
    Ey_phi    = _E_y_phi_cond(mu_cond, sig2_cond)
    phip1     = np.sqrt(2.0/np.pi) * np.exp(-0.5*(m1+_ZZ)**2)
    return float(np.dot(_WW, phip1 * Ey_phi))


# ── Array-vectorised H_mu and H_mu_prime ─────────────────────────────────────
# Accept q, m1, m2 as numpy arrays of any shape; broadcast over quadrature axis.

def dH_mu_dm2_arr(q, m1, m2):
    """E[phi(m1+G1)*phi'(m2+G2)], broadcast over arbitrary-shaped arrays q, m1, m2.
    This is partial derivative of H_mu w.r.t. the second mean m2.
    Used for the nu_w signal terms (Fix C): derivative of H_ss,d and H_cc,d
    w.r.t. the source-time expert-teacher overlap P_self(tau).
    """
    q  = np.clip(np.asarray(q,  float), -1+1e-10, 1-1e-10)
    m1 = np.asarray(m1, float); m2 = np.asarray(m2, float)
    if q.ndim == 0:
        return dH_mu_dm2(float(q), float(m1), float(m2))
    ZZ    = _ZZ.reshape((-1,) + (1,)*q.ndim)
    WW    = _WW.reshape((-1,) + (1,)*q.ndim)
    denom = np.sqrt(2. - q*q)
    phi1  = erf((m1 + ZZ) / np.sqrt(2.))
    phip2 = np.sqrt(2./np.pi) * np.exp(-(m2 + q*ZZ)**2 / (2.*(2.-q*q))) / denom
    return np.sum(WW * phi1 * phip2, axis=0)


def H_mu_arr(q, m1, m2):
    """H_mu broadcast over arbitrary-shaped arrays q, m1, m2.

    Scalar fallback delegates to the standard H_mu (same computation).
    Array path reshapes the quadrature axis to broadcast over any shape.
    """
    q  = np.clip(np.asarray(q,  float), -1+1e-10, 1-1e-10)
    m1 = np.asarray(m1, float); m2 = np.asarray(m2, float)
    if q.ndim == 0:
        return H_mu(float(q), float(m1), float(m2))   # scalar path
    # Broadcast quadrature axis (K,) over result shape (*S)
    ZZ = _ZZ.reshape((-1,) + (1,)*q.ndim)
    WW = _WW.reshape((-1,) + (1,)*q.ndim)
    phi1   = erf((m1 + ZZ) / np.sqrt(2.))
    denom2 = np.sqrt(2.*(2. - q*q))
    phi2   = erf((m2 + q*ZZ) / denom2)
    return np.sum(WW * phi1 * phi2, axis=0)


def H_mu_prime_arr(q, m1, m2):
    """H_mu_prime broadcast over arbitrary-shaped arrays q, m1, m2."""
    q  = np.clip(np.asarray(q,  float), -1+1e-10, 1-1e-10)
    m1 = np.asarray(m1, float); m2 = np.asarray(m2, float)
    if q.ndim == 0:
        return H_mu_prime(float(q), float(m1), float(m2))   # scalar path
    ZZ    = _ZZ.reshape((-1,) + (1,)*q.ndim)
    WW    = _WW.reshape((-1,) + (1,)*q.ndim)
    denom = np.sqrt(2. - q*q)
    phip1 = np.sqrt(2./np.pi) * np.exp(-0.5*(m1+ZZ)**2)
    phip2 = np.sqrt(2./np.pi) * np.exp(-(m2+q*ZZ)**2 / (2.*(2.-q*q))) / denom
    return np.sum(WW * phip1 * phip2, axis=0)


# ── Self-consistency correction kernels (t=0 boundary term) ──────────────────
# At t=0 the DMFT signal uses the pure teacher kernel E[G·y·σ'(z^w)·λ].  The
# GF, however, computes residuals r = y − f₀ where f₀ = G·ā₀·σ(z^w) ≠ 0.
# The leading-order correction is:
#
#   δΦ_self = ā₀ · G_s · K_phi_phip_lam(P_self, κ·P_self, κ)
#   δΦ_cross = ā₀ · G_{co} · K_cross_sc(P_self, P_cross, κ)
#
# where G_s = E_{c=e}[g_e²] and G_{co} = E_{c≠e}[g_e·g_{e'}] = a₀² − a₁²(1+κ²rD²).
# These are subtracted from tPhi_s[0] / tPhi_c[0] at the start of run().

def K_phi_phip_lam(q, m1, m2):
    """E[phi(m1+G1) · phi'(m1+G1) · (m2+G2)],  Cov(G1,G2) = q.

    SC correction for P_self ODE: when f₀ = G·ā₀·phi(z^w), the correction
    to the signal kernel is ā₀·G_s·K_phi_phip_lam(P_self, κ·P_self, κ).

    Derivation: condition on G1, so E[G2|G1=z] = q·z, giving
        K = m2·E[phi·phi'(m1+G)] + q·E[G1·phi·phi'(m1+G)]
    Both terms are 1-D Gauss–Hermite integrals over G1 ∼ N(0,1).

    At m1=0 this is zero by parity (phi·phi' is an odd function).
    """
    q   = float(np.clip(q, -1+1e-10, 1-1e-10))
    z   = m1 + _ZZ                                               # m1+G1 nodes
    pp  = erf(z/np.sqrt(2.)) * np.sqrt(2./np.pi) * np.exp(-0.5*z**2)  # phi·phi'
    I   = float(np.dot(_WW, pp))         # E[phi·phi'(m1+G)]
    J   = float(np.dot(_WW, _ZZ * pp))  # E[G1·phi·phi'(m1+G)]   (G1 not m1+G1)
    return m2 * I + q * J