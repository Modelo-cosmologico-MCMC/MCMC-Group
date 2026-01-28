"""Módulo Lattice-Gauge para el MCMC.

Este módulo implementa simulaciones de teoría gauge en retícula
para explorar el mass gap ontológico del MCMC.

En el MCMC, el mass gap no es un parámetro fenomenológico, sino
un evento ontológico:
    - Entre S=0.010 y S=1.000 la masa primordial Mp se aniquila
    - En S=1.000 se sella la fase volumétrica V₃D
    - En S=1.001 la métrica V₃₊₁D se estabiliza

El mass gap mínimo: E_min = k · ΔS, donde k = M_Pl·c², ΔS = 10⁻³
"""
from .wilson import WilsonAction as WilsonAction
from .wilson import WilsonActionParams as WilsonActionParams
from .beta_of_S import beta_of_S as beta_of_S
from .beta_of_S import BetaParams as BetaParams
from .monte_carlo import MetropolisSampler as MetropolisSampler
from .monte_carlo import MCParams as MCParams
from .mass_gap import MassGapExtractor as MassGapExtractor
from .mass_gap import extract_mass_gap as extract_mass_gap
from .mass_gap import run_S_scan as run_S_scan

__all__ = [
    "WilsonAction",
    "WilsonActionParams",
    "beta_of_S",
    "BetaParams",
    "MetropolisSampler",
    "MCParams",
    "MassGapExtractor",
    "extract_mass_gap",
    "run_S_scan",
]
