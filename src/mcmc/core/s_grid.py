"""
Módulo de Rejilla Entrópica S y Sellos Ontológicos
===================================================

Este módulo implementa la rejilla entrópica discreta S con paso ΔS=10^{-3}
y los sellos ontológicos (nodos críticos) del MCMC.

La evolución en el MCMC no se parametriza primariamente por t sino por la
variable entrópica S, que define los "bloques" de la infraestructura computacional.

Referencias:
    - Tratado MCMC Maestro, Bloque 0 y Bloque 1 (núcleo ontológico)
    - Apartado Computacional: rejilla y validación de sellos
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import numpy as np


@dataclass(frozen=True)
class Seals:
    """
    Sellos Ontológicos del MCMC.

    Los sellos son nodos críticos en la variable entrópica S donde ocurren
    transiciones fundamentales en el modelo:

    - S1 = 0.010: Inicio del régimen ontológico (post-Bloque 0)
    - S2 = 0.100: Transición temprana (régimen rígido)
    - S3 = 1.000: Transición tardía (aproximación al sello final)
    - S4 = 1.001: Sello final de normalización (a=1, z=0, hoy)

    Estos valores provienen del núcleo ontológico computacional (Bloque 1).
    """
    S1: float = 0.010   # Primer sello: inicio post-geométrico
    S2: float = 0.100   # Segundo sello: transición temprana
    S3: float = 1.000   # Tercer sello: transición tardía
    S4: float = 1.001   # Sello final: normalización (hoy)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Retorna los sellos como tupla ordenada."""
        return (self.S1, self.S2, self.S3, self.S4)

    def as_array(self) -> np.ndarray:
        """Retorna los sellos como array numpy."""
        return np.array([self.S1, self.S2, self.S3, self.S4])

    def get_all(self) -> dict:
        """Retorna diccionario con todos los sellos."""
        return {"S1": self.S1, "S2": self.S2, "S3": self.S3, "S4": self.S4}


@dataclass
class SGrid:
    """
    Rejilla Entrópica S del MCMC.

    Define la malla discreta sobre la cual se integran las ecuaciones de fondo
    y se calculan los canales energéticos. La rejilla típica tiene:

    - S ∈ [0.010, 1.001]
    - N_S ≈ 991 nodos
    - ΔS = 10^{-3}

    Attributes:
        S_min: Valor mínimo de S (coincide con S1)
        S_max: Valor máximo de S (coincide con S4)
        dS: Paso de la rejilla
        seals: Objeto Seals con los nodos críticos

    Example:
        >>> grid = SGrid()
        >>> S = grid.build()
        >>> print(f"Nodos: {len(S)}, S_min={S[0]:.3f}, S_max={S[-1]:.3f}")
        Nodos: 992, S_min=0.010, S_max=1.001
    """
    S_min: float = 0.010
    S_max: float = 1.001
    dS: float = 1e-3
    seals: Seals = field(default_factory=Seals)

    def __post_init__(self):
        """Validación post-inicialización."""
        if self.S_min != self.seals.S1:
            raise ValueError(f"S_min ({self.S_min}) debe coincidir con S1 ({self.seals.S1})")
        if self.S_max != self.seals.S4:
            raise ValueError(f"S_max ({self.S_max}) debe coincidir con S4 ({self.seals.S4})")
        if self.dS <= 0:
            raise ValueError(f"dS debe ser positivo, se recibió {self.dS}")

    def build(self) -> np.ndarray:
        """
        Construye la rejilla entrópica S.

        Returns:
            Array numpy con los valores de S desde S_min hasta S_max
            con paso dS. El último punto está garantizado en la rejilla.
        """
        S = np.arange(self.S_min, self.S_max + 0.5 * self.dS, self.dS)
        # Asegurar que el último punto sea exactamente S_max
        if not np.isclose(S[-1], self.S_max, atol=1e-12):
            S = np.append(S, self.S_max)
        return S

    def build_with_forced_seals(self) -> np.ndarray:
        """
        Construye la rejilla asegurando que todos los sellos están exactamente en ella.

        Si algún sello no cae exactamente en la rejilla regular, se ajusta
        el punto más cercano para que coincida con el sello.

        Returns:
            Array numpy con la rejilla que incluye todos los sellos exactamente.
        """
        S = self.build()
        for seal_val in self.seals.as_array():
            # Encontrar el índice más cercano
            idx = np.argmin(np.abs(S - seal_val))
            S[idx] = seal_val
        return np.sort(np.unique(S))

    def assert_seals_on_grid(self, S: np.ndarray) -> None:
        """
        Verifica que todos los sellos están exactamente en la rejilla.

        Esta verificación es parte de la validación ontológica del Bloque 1
        y es precondición para ejecutar inferencia o N-body.

        Args:
            S: Array con la rejilla a verificar

        Raises:
            ValueError: Si algún sello no está en la rejilla
        """
        seal_dict = self.seals.get_all()
        for name, val in seal_dict.items():
            if not np.any(np.isclose(S, val, atol=1e-12)):
                raise ValueError(
                    f"Sello {name}={val} no está exactamente en la rejilla. "
                    f"Use build_with_forced_seals() para garantizar inclusión."
                )

    def get_seal_indices(self, S: np.ndarray) -> dict:
        """
        Obtiene los índices de los sellos en la rejilla.

        Args:
            S: Array con la rejilla

        Returns:
            Diccionario {nombre_sello: índice}
        """
        indices = {}
        seal_dict = self.seals.get_all()
        for name, val in seal_dict.items():
            idx = np.argmin(np.abs(S - val))
            if not np.isclose(S[idx], val, atol=1e-12):
                raise ValueError(f"Sello {name}={val} no encontrado en la rejilla")
            indices[name] = idx
        return indices

    @property
    def n_nodes(self) -> int:
        """Número aproximado de nodos en la rejilla."""
        return int((self.S_max - self.S_min) / self.dS) + 1

    def get_regime_mask(self, S: np.ndarray, regime: str) -> np.ndarray:
        """
        Obtiene máscara booleana para un régimen específico.

        Args:
            S: Array con la rejilla
            regime: Uno de 'early' (S1-S2), 'middle' (S2-S3), 'late' (S3-S4)

        Returns:
            Máscara booleana del régimen solicitado
        """
        seals = self.seals
        if regime == 'early':
            return (S >= seals.S1) & (S < seals.S2)
        elif regime == 'middle':
            return (S >= seals.S2) & (S < seals.S3)
        elif regime == 'late':
            return (S >= seals.S3) & (S <= seals.S4)
        else:
            raise ValueError(f"Régimen desconocido: {regime}. Use 'early', 'middle' o 'late'")


def create_default_grid() -> Tuple[SGrid, np.ndarray]:
    """
    Función de conveniencia para crear la rejilla por defecto.

    Returns:
        Tupla (objeto SGrid, array S)
    """
    grid = SGrid()
    S = grid.build_with_forced_seals()
    grid.assert_seals_on_grid(S)
    return grid, S


# Constantes exportadas para uso rápido
DEFAULT_SEALS = Seals()
DEFAULT_DS = 1e-3
DEFAULT_S_MIN = 0.010
DEFAULT_S_MAX = 1.001
