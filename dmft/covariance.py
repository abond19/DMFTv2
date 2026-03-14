"""
Covariance functions for two-layer network DMFT.

For activation σ : ℝ → ℝ (zero-mean), define
  h(q)      = E[σ(G₁)σ(G₂)]   where Cov(G₁,G₂) = q
  φ̂(v)     = E[σ(G) · v·G]    (k=1 single-index projection score)
  ‖φ‖²      = h(1)

h̃(q) = h(q) − h′(0)·q  removes the linear part (lazy regime, Section F).

References: Section A (Eqs A.4–A.7), Section C.4 (Eq C.42), Eq F.24.
"""

from typing import Callable, NamedTuple


class CovarianceFunctions(NamedTuple):
    """All covariance-derived callables needed by the DMFT solver."""
    h:            Callable   # h(q)
    h_p:          Callable   # h′(q)
    h_pp:         Callable   # h″(q)
    phi_hat:      Callable   # φ̂(v)
    phi_hat_grad: Callable   # dφ̂/dv
    h_tilde:      Callable   # h̃(q) = h(q) − h′(0)·q
    h_tilde_p:    Callable   # h̃′(q)
    h_tilde_pp:   Callable   # h̃″(q)
    phi_norm_sq:  float      # ‖φ‖² = h(1)
    h_prime_0:    float      # h′(0)


def from_polynomial(
    coeffs_h: dict,
    coeffs_phi_hat: dict | None = None,
    phi_norm_sq: float = 0.0,
) -> CovarianceFunctions:
    """
    Build CovarianceFunctions from polynomial coefficients.

    Args:
        coeffs_h:       {power: coeff} dict for h(q) = Σ cₖ qᵏ
        coeffs_phi_hat: {power: coeff} dict for φ̂(v), or None (pure noise)
        phi_norm_sq:    ‖φ‖² = h(1) = Σ cₖ
    """
    h_pw = sorted(coeffs_h.keys())
    h_co = [coeffs_h[k] for k in h_pw]

    def h(q):
        return sum(c * q**k for k, c in zip(h_pw, h_co))

    def h_p(q):
        return sum(c * k * q**(k-1) for k, c in zip(h_pw, h_co) if k >= 1)

    def h_pp(q):
        return sum(c * k*(k-1) * q**(k-2) for k, c in zip(h_pw, h_co) if k >= 2)

    hp0 = sum(c for k, c in zip(h_pw, h_co) if k == 1)

    def h_tilde(q):    return h(q) - hp0 * q
    def h_tilde_p(q):  return h_p(q) - hp0
    def h_tilde_pp(q): return h_pp(q)

    if coeffs_phi_hat is None:
        phi_hat      = lambda v: 0.0
        phi_hat_grad = lambda v: 0.0
    else:
        ph_pw = sorted(coeffs_phi_hat.keys())
        ph_co = [coeffs_phi_hat[k] for k in ph_pw]

        def phi_hat(v):
            return sum(c * v**k for k, c in zip(ph_pw, ph_co))

        def phi_hat_grad(v):
            return sum(c * k * v**(k-1) for k, c in zip(ph_pw, ph_co) if k >= 1)

    return CovarianceFunctions(
        h=h, h_p=h_p, h_pp=h_pp,
        phi_hat=phi_hat, phi_hat_grad=phi_hat_grad,
        h_tilde=h_tilde, h_tilde_p=h_tilde_p, h_tilde_pp=h_tilde_pp,
        phi_norm_sq=phi_norm_sq, h_prime_0=hp0,
    )


# ── Named paper configurations ────────────────────────────────────────────────

def paper_config_1() -> CovarianceFunctions:
    """h(q) = φ̂(q) = (9/10)q + q²/2.  Figs 3, 13–17."""
    c = {1: 9.0/10, 2: 1.0/2}
    return from_polynomial(c, c, phi_norm_sq=sum(c.values()))


def paper_config_2() -> CovarianceFunctions:
    """h(q) = φ̂(q) = (9/10)q + q³/6.  Figs 1, 2, 4, 5, 18–31."""
    c = {1: 9.0/10, 3: 1.0/6}
    return from_polynomial(c, c, phi_norm_sq=sum(c.values()))


def paper_config_3() -> CovarianceFunctions:
    """h(q) = (3/10)q + q²/2, pure noise.  Figs 8–10, 32–33."""
    return from_polynomial({1: 3.0/10, 2: 1.0/2}, None, phi_norm_sq=0.0)


def paper_config_4() -> CovarianceFunctions:
    """h(q) = (9/10)q + q²/2, pure noise.  Figs 11–12."""
    return from_polynomial({1: 9.0/10, 2: 1.0/2}, None, phi_norm_sq=0.0)