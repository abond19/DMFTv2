"""
dmft — SymmDMFT solver for two-layer network training dynamics.

Typical usage
─────────────
    from dmft.config import fig1_config
    from dmft.solver import run_dmft
    from dmft.observables import get_observables

    config = fig1_config(m=64)
    state  = run_dmft(config)
    obs    = get_observables(state, config.cov, config.m, config.data.tau)
"""

from dmft.covariance import (
    CovarianceFunctions, from_polynomial,
    paper_config_1, paper_config_2, paper_config_3, paper_config_4,
)
from dmft.config import (
    DataConfig, InitConfig, SolverConfig, ExperimentConfig,
    fig1_config, fig2_config, fig3_config, fig4_config,
)
from dmft.state import SymmDMFTState, create_symm_state
from dmft.solver import run_dmft
from dmft.observables import get_observables, train_error, test_error

__all__ = [
    "CovarianceFunctions", "from_polynomial",
    "paper_config_1", "paper_config_2", "paper_config_3", "paper_config_4",
    "DataConfig", "InitConfig", "SolverConfig", "ExperimentConfig",
    "fig1_config", "fig2_config", "fig3_config", "fig4_config",
    "SymmDMFTState", "create_symm_state",
    "run_dmft",
    "get_observables", "train_error", "test_error",
]