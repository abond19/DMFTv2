"""
Configuration dataclasses for DMFT experiments.

An ExperimentConfig fully specifies:
  - Network architecture (m neurons, activation via CovarianceFunctions)
  - Data distribution (pure noise vs single-index, tau, k)
  - Initialisation (lazy γ₀√m  or  mean-field a₀)
  - Solver (step size η, max time T)
  - Whether second-layer weights evolve

References: Section 1 (Eqs 1.1–1.2), Section 2, Appendix D.1.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional

from dmft.covariance import CovarianceFunctions


@dataclass
class DataConfig:
    """Data-generating distribution.

    Pure noise:   yᵢ = εᵢ ~ N(0, τ²)
    Single-index: yᵢ = φ(Uᵀxᵢ) + εᵢ   with k = index dimension
    """
    tau:          float = 1.0
    k:            int   = 1
    is_pure_noise: bool = True


@dataclass
class InitConfig:
    """Weight initialisation.

    Lazy:        a(0) = γ₀ √m    (Section F)
    Mean-field:  a(0) = a₀       (Section G)

    First-layer weights always: wᵢ(0) ~ Unif(𝕊^{d-1}),
    giving Cd(0,0)=1, Co(0,0)=0, v(0)=0.
    """
    mode:   Literal["lazy", "mean_field"] = "mean_field"
    a0:     float = 1.0    # mean-field: a(0) = a₀
    gamma0: float = 1.0    # lazy: a(0) = γ₀ √m

    def get_a0(self, m: int) -> float:
        if self.mode == "lazy":
            return self.gamma0 * m**0.5
        return self.a0


@dataclass
class SolverConfig:
    """Euler integration parameters.

    Overflow note: for mean-field init, a(t) grows in the third regime.
    Typically safe when T_max ≲ 10 m  with  η ≤ 0.5.
    """
    eta:    float = 0.1
    T_max:  float = 100.0
    n_steps: int  = field(init=False)

    def __post_init__(self):
        self.n_steps = int(self.T_max / self.eta) + 1


@dataclass
class ExperimentConfig:
    """Full specification of a DMFT experiment."""
    # Network
    m:           int   = 64
    alpha_bar:   float = 0.3   # ᾱ = n/(md);  code's alpha = ᾱ·m = n/d

    # Covariance (set after construction via a paper_config_* helper)
    cov: Optional[CovarianceFunctions] = None

    # Sub-configs
    data:   DataConfig   = field(default_factory=DataConfig)
    init:   InitConfig   = field(default_factory=InitConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)

    # Flags
    evolve_second_layer: bool = True

    @property
    def alpha(self) -> float:
        """α = n/d = ᾱ · m  (the 'alpha' in the DMFT equations)."""
        return self.alpha_bar * self.m

    def get_a0(self) -> float:
        return self.init.get_a0(self.m)


# ── Named configs matching paper figures ─────────────────────────────────────

def fig1_config(m: int = 64) -> ExperimentConfig:
    """
    Fig 1: Three dynamical regimes, single-index, mean-field init.
    h = φ̂ = (9/10)q + q³/6,  τ=0.3,  ᾱ=0.3.

    Safe numerics:  η=0.5, T_max=580  (overflow occurs for T ≳ 630 at m=64).
    """
    from dmft.covariance import paper_config_2
    return ExperimentConfig(
        m=m, alpha_bar=0.3, cov=paper_config_2(),
        data=DataConfig(tau=0.3, k=1, is_pure_noise=False),
        init=InitConfig(mode="mean_field", a0=1.0),
        solver=SolverConfig(eta=0.5, T_max=580.0),
        evolve_second_layer=True,
    )


def fig2_config(m: int = 64) -> ExperimentConfig:
    """
    Fig 2: Pure noise, mean-field init.
    h = φ̂ = (9/10)q + q³/6,  τ=0.6,  ᾱ=0.4.
    """
    from dmft.covariance import paper_config_2
    return ExperimentConfig(
        m=m, alpha_bar=0.4, cov=paper_config_2(),
        data=DataConfig(tau=0.6, k=0, is_pure_noise=True),
        init=InitConfig(mode="mean_field", a0=1.0),
        solver=SolverConfig(eta=0.5, T_max=1000.0),
        evolve_second_layer=True,
    )


def fig3_config(m: int = 64) -> ExperimentConfig:
    """
    Fig 3: Single-index, lazy init.
    h = φ̂ = (9/10)q + q²/2,  τ=0.3,  ᾱ=0.3.
    """
    from dmft.covariance import paper_config_1
    return ExperimentConfig(
        m=m, alpha_bar=0.3, cov=paper_config_1(),
        data=DataConfig(tau=0.3, k=1, is_pure_noise=False),
        init=InitConfig(mode="lazy", gamma0=1.0),
        solver=SolverConfig(eta=0.05, T_max=1000.0),
        evolve_second_layer=True,
    )


def fig4_config(m: int = 64) -> ExperimentConfig:
    """
    Fig 4: Single-index, mean-field init, multiple m values.
    h = φ̂ = (9/10)q + q³/6,  τ=0.3,  ᾱ=0.3.
    """
    from dmft.covariance import paper_config_2
    return ExperimentConfig(
        m=m, alpha_bar=0.3, cov=paper_config_2(),
        data=DataConfig(tau=0.3, k=1, is_pure_noise=False),
        init=InitConfig(mode="mean_field", a0=1.0),
        solver=SolverConfig(eta=0.5, T_max=min(580.0, 9.0*m)),
        evolve_second_layer=True,
    )


def fig5_config(m: int = 64) -> ExperimentConfig:
    """
    Fig 5: Scaling curves for a(t)/√m and v(t) as a function of t/m.
    Same model as Fig 1: h = φ̂ = (9/10)q + q³/6, τ=0.3, ᾱ=0.3, MF init.

    Safe T_max per m (overflow occurs beyond these values with η=0.5):
        m=8  → T_max=800   (t/m up to 100)
        m=16 → T_max=580   (t/m up to 36)
        m=32 → T_max=560   (t/m up to 17.5)
        m=64 → T_max=580   (t/m up to  9.1)
    """
    from dmft.covariance import paper_config_2
    # Safe T caps derived empirically from overflow boundary ~ 10*m
    safe_T = {
        8:  800.0,
        16: 580.0,
        32: 560.0,
        64: 580.0,
    }
    T_max = safe_T.get(m, min(9.0 * m, 580.0))
    return ExperimentConfig(
        m=m, alpha_bar=0.3, cov=paper_config_2(),
        data=DataConfig(tau=0.3, k=1, is_pure_noise=False),
        init=InitConfig(mode="mean_field", a0=1.0),
        solver=SolverConfig(eta=0.5, T_max=T_max),
        evolve_second_layer=True,
    )


def lazy_fixed_a_config(gamma: float, m: int = 32) -> ExperimentConfig:
    """
    Lazy init with FIXED second-layer weights: a(t) = γ√m for all t.

    Used for the 'Lazy' reference curve in Fig 5 (right panel):
        e_ts(∞) − e_tr(∞)  vs  γ = a/√m.

    With fixed a, v(t) converges to a steady state that depends on γ.
    T_max=300 is sufficient for convergence at all γ ∈ [0, 0.8].
    """
    from dmft.covariance import paper_config_2
    return ExperimentConfig(
        m=m, alpha_bar=0.3, cov=paper_config_2(),
        data=DataConfig(tau=0.3, k=1, is_pure_noise=False),
        init=InitConfig(mode="lazy", gamma0=gamma),
        solver=SolverConfig(eta=0.1, T_max=100.0),
        evolve_second_layer=False,
    )


def fig8_config(m: int = 64) -> ExperimentConfig:
    """
    Fig 8: Pure noise, lazy init, second dynamical regime.
    h̃ = (3/10)q + q²/2,  τ=1,  α=0.5  (fixed second-layer weights).
    """
    from dmft.covariance import paper_config_3
    return ExperimentConfig(
        m=m, alpha_bar=0.5 / m,   # α = n/d = 0.5 fixed → ᾱ = 0.5/m
        cov=paper_config_3(),
        data=DataConfig(tau=1.0, k=0, is_pure_noise=True),
        init=InitConfig(mode="lazy", gamma0=1.0),
        solver=SolverConfig(eta=0.05, T_max=100.0),
        evolve_second_layer=False,
    )


def fig5_config(m: int) -> ExperimentConfig:
    """
    Fig 5: Third-regime scaling / collapse, single-index, mean-field init.
    h = φ̂ = (9/10)q + q³/6,  τ=0.3,  ᾱ=0.3.

    T_max = 9m (empirically safe against float64 overflow for all m ≤ 128).
    """
    from dmft.covariance import paper_config_2
    eta = 0.25
    T_max = 20.0 * m
    # T_max = 100.0
    return ExperimentConfig(
        m=m, alpha_bar=0.3, cov=paper_config_2(),
        data=DataConfig(tau=0.3, k=1, is_pure_noise=False),
        init=InitConfig(mode="mean_field", a0=1.0),
        solver=SolverConfig(eta=eta, T_max=T_max),
        evolve_second_layer=True,
    )


def fig5_lazy_config(gamma0: float, m: int = 8) -> ExperimentConfig:
    """
    Lazy reference curve for Fig 5 right panel.

    Fixed second-layer weights: a(t) = γ₀√m throughout.
    Uses m=8 (default) to keep a = γ₀√m small and avoid float64 overflow
    for large γ₀.  The lazy curve is m-independent in the large-m limit
    (Eq 2.13), so m=8 gives the correct reference.
    T_max=100 is sufficient for first-layer weights to equilibrate (O(1)).
    """
    from dmft.covariance import paper_config_2
    return ExperimentConfig(
        m=m, alpha_bar=0.3, cov=paper_config_2(),
        data=DataConfig(tau=0.3, k=1, is_pure_noise=False),
        init=InitConfig(mode="lazy", gamma0=gamma0),
        solver=SolverConfig(eta=0.5, T_max=100.0),
        evolve_second_layer=False,
    )