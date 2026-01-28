"""Módulo de crecimiento de estructuras para el MCMC.

Implementa:
    - Crecimiento lineal D(S), D(z) en variable entrópica
    - Predicción fσ₈(z) para comparación con RSD
    - Gravedad modificada μ(a), η(a)
"""
from .linear_growth import (
    GrowthParams as GrowthParams,
    LinearGrowthSolver as LinearGrowthSolver,
    D_of_z_LCDM as D_of_z_LCDM,
    f_of_z_LCDM as f_of_z_LCDM,
)
from .f_sigma8 import (
    FSigma8Predictor as FSigma8Predictor,
    compute_f_sigma8 as compute_f_sigma8,
    f_sigma8_LCDM as f_sigma8_LCDM,
)
from .mu_eta import (
    MuEtaParams as MuEtaParams,
    MuEtaFromS as MuEtaFromS,
    mu_CPL as mu_CPL,
    eta_CPL as eta_CPL,
    Sigma_lensing as Sigma_lensing,
    Upsilon_RSD as Upsilon_RSD,
    solve_growth_modified as solve_growth_modified,
    compare_modified_gravity as compare_modified_gravity,
)

__all__ = [
    "GrowthParams",
    "LinearGrowthSolver",
    "D_of_z_LCDM",
    "f_of_z_LCDM",
    "FSigma8Predictor",
    "compute_f_sigma8",
    "f_sigma8_LCDM",
    "MuEtaParams",
    "MuEtaFromS",
    "mu_CPL",
    "eta_CPL",
    "Sigma_lensing",
    "Upsilon_RSD",
    "solve_growth_modified",
    "compare_modified_gravity",
]
