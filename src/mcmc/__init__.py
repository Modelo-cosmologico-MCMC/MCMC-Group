"""Modelo Cosmológico de Múltiples Colapsos (MCMC).

Este paquete implementa el MCMC, un modelo cosmológico basado en la
dualidad Masa Primordial / Espacio Primordial (Mp/Ep) y la variable
entrópica S.

Módulos principales:
    - core: Funciones centrales (Ley de Cronos, Friedmann, s_grid)
    - channels: Canales de energía (ρ_id, ρ_lat)
    - observables: Observables cosmológicos (H(z), SNe, BAO)
    - nbody: Simulaciones N-body Cronos
    - lattice: Simulaciones lattice-gauge para mass gap
    - inference: Ajuste bayesiano con emcee
"""
__all__ = ["__version__"]
__version__ = "0.1.0"
