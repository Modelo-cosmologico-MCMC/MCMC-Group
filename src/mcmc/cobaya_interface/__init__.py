"""Componente de fondo MCMC con interfaz estilo Cobaya.

Este paquete implementa la Prioridad 1 del plan de validacion:
un componente teorico que recibe parametros MCMC y devuelve
H(z), D_M, D_H, D_V y D_L, recuperando LCDM en un limite verificable.

Modulos:
- background: BackgroundCalculator (motor numerico, sin dependencia de cobaya)
- lcdm_reference: implementacion LCDM independiente (quad) para verificacion
- theory: wrapper cobaya.theory.Theory (requiere cobaya instalado, opcional)
"""
from mcmc.cobaya_interface.background import (
    BackgroundCalculator,
    BackgroundParams,
)
from mcmc.cobaya_interface.lcdm_reference import LCDMReference

__all__ = [
    "BackgroundCalculator",
    "BackgroundParams",
    "LCDMReference",
]
