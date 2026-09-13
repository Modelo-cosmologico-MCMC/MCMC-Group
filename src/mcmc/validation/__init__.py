"""Validacion del pipeline: datos sinteticos y recuperacion de parametros.

Prioridad 2 del plan de validacion: generar BAO y SNe simulados con
parametros conocidos, recuperarlos con el mismo pipeline de inferencia
que se usara con datos reales, y detectar degeneraciones o errores del
mapeo S <-> z antes de tocar datos observacionales.
"""
from mcmc.validation.synthetic import SyntheticSpec, generate_synthetic_datasets
from mcmc.validation.recovery import (
    RecoveryResult,
    fit_map,
    run_recovery,
    s_map_roundtrip_check,
)

__all__ = [
    "SyntheticSpec",
    "generate_synthetic_datasets",
    "RecoveryResult",
    "fit_map",
    "run_recovery",
    "s_map_roundtrip_check",
]
