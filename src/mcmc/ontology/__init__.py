"""Ontología del MCMC: Mapa entrópico, Campo de Adrián, Métrica Dual.

Este submódulo implementa la estructura ontológica del MCMC:

- s_map: Mapa completo S↔z↔t↔a con Ley de Cronos
- adrian_field: Campo de Adrián Φ_Ad con fases escalar y tensorial
- dual_metric: Métrica Dual Relativa g_μν(S)
"""
from .s_map import (
    SMapParams,
    EntropyMap,
    create_default_map,
    S_of_z_simple,
    z_of_S_simple,
)

from .adrian_field import (
    AdrianFieldParams,
    TransitionParams,
    AdrianField,
)

from .dual_metric import (
    MDRParams,
    DualRelativeMetric,
    create_LCDM_metric,
    create_MCMC_metric,
)

__all__ = [
    # s_map
    "SMapParams",
    "EntropyMap",
    "create_default_map",
    "S_of_z_simple",
    "z_of_S_simple",
    # adrian_field
    "AdrianFieldParams",
    "TransitionParams",
    "AdrianField",
    # dual_metric
    "MDRParams",
    "DualRelativeMetric",
    "create_LCDM_metric",
    "create_MCMC_metric",
]
