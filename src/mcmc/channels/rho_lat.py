"""Canal Latente ρ_lat(S) - Energía sellada y decaimiento entrópico.

CORRECCIÓN ONTOLÓGICA (2025): S ∈ [0, 100]
- Pre-geométrico: S ∈ [0, 1.001) - Canal latente no activo
- Post-Big Bang: S ∈ [1.001, 95.07] - Decaimiento entrópico activo

El canal latente representa la Masa Cuántica Virtual (MCV): tensión sellada
que no se manifiesta como materia convencional. Decae por conversión
entrópica (no dilución adiabática) y su masa efectiva se invierte en
creación de espacio, generando tiempo relativo.

Ecuaciones fundamentales:
    dρ_lat/dS = -κ_lat(S) · ρ_lat
    κ_lat(S) = κ₀ · tanh[(S - S_★) / ΔS_★]

Ontology:
    Mp → (bariones + halos) + MCV[ρ_lat] + ECV[ρ_id]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.integrate import quad, cumulative_trapezoid


@dataclass
class LatentChannelParams:
    """Parámetros del canal latente ρ_lat(S).

    CORRECCIÓN: S ∈ [0, 100]
    - S_star = 1.001 corresponde al Big Bang
    - Post-Big Bang: S ∈ [1.001, 95.07]

    Attributes:
        enabled: Si el canal está activo
        kappa_0: Coeficiente de decaimiento base
        S_star: Punto de encendido del decaimiento (Big Bang = 1.001)
        dS_star: Anchura de transición (mayor para nuevo rango)
        rho_lat_star: Densidad latente de referencia en unidades de ρ_c
        beta: Coeficiente de acople al tiempo relativo
    """
    enabled: bool = True
    kappa_0: float = 0.01          # Reducido para rango más amplio
    S_star: float = 1.001          # Big Bang
    dS_star: float = 1.0           # Anchura ajustada para S ∈ [1, 95]
    rho_lat_star: float = 0.01     # Fracción de ρ_c
    beta: float = 1.0


class LatentChannel:
    """Implementación del canal latente ρ_lat(S).

    El canal latente completa la descomposición de energía del MCMC,
    representando la tensión sellada en el espacio. Actúa como
    "colchón tensional" en regiones centrales de halos.

    Attributes:
        params: Parámetros del canal latente
    """

    def __init__(self, params: LatentChannelParams | None = None):
        """Inicializa el canal latente.

        Args:
            params: Parámetros. Si None, usa valores por defecto.
        """
        self.params = params or LatentChannelParams()

    def kappa_lat(self, S: float | np.ndarray) -> float | np.ndarray:
        """Coeficiente de decaimiento entrópico κ_lat(S).

        κ_lat(S) = κ₀ · tanh[(S - S_★) / ΔS_★]

        Args:
            S: Variable entrópica

        Returns:
            Coeficiente de decaimiento
        """
        if not self.params.enabled:
            return np.zeros_like(S) if isinstance(S, np.ndarray) else 0.0

        p = self.params
        arg = (np.asarray(S) - p.S_star) / max(p.dS_star, 1e-12)
        return p.kappa_0 * np.tanh(arg)

    def _integral_kappa(self, S: float) -> float:
        """Integral de κ_lat desde S_star hasta S.

        ∫_{S_★}^{S} κ_lat(S') dS'
        """
        if S <= self.params.S_star:
            return 0.0
        result, _ = quad(self.kappa_lat, self.params.S_star, S)
        return result

    def rho_lat(self, S: float | np.ndarray) -> float | np.ndarray:
        """Densidad latente en función de S.

        ρ_lat(S) = ρ_lat^★ · exp[-∫ κ_lat(S') dS']

        Args:
            S: Variable entrópica

        Returns:
            Densidad latente en unidades de ρ_c
        """
        if not self.params.enabled:
            return np.zeros_like(S) if isinstance(S, np.ndarray) else 0.0

        S_arr = np.atleast_1d(S)
        rho = np.zeros_like(S_arr, dtype=float)

        for i, s in enumerate(S_arr):
            integral = self._integral_kappa(float(s))
            rho[i] = self.params.rho_lat_star * np.exp(-integral)

        return rho[0] if np.ndim(S) == 0 else rho

    def rho_lat_array(self, S_array: np.ndarray) -> np.ndarray:
        """Calcula ρ_lat para un array de S de forma eficiente.

        Usa integración acumulativa para arrays ordenados.

        Args:
            S_array: Array de valores de S (debe estar ordenado)

        Returns:
            Array de ρ_lat
        """
        if not self.params.enabled:
            return np.zeros_like(S_array)

        # Calcular κ_lat para todo el array
        kappa = self.kappa_lat(S_array)

        # Integración acumulativa
        if len(S_array) < 2:
            return np.array([self.params.rho_lat_star])

        integral = cumulative_trapezoid(kappa, S_array, initial=0.0)

        # Ajustar inicio desde S_star
        mask = S_array >= self.params.S_star
        integral_adjusted = np.zeros_like(integral)
        if np.any(mask):
            idx_start = np.argmax(mask)
            integral_adjusted[idx_start:] = integral[idx_start:] - integral[idx_start]

        return self.params.rho_lat_star * np.exp(-integral_adjusted)

    def delta_lat(self, S: float | np.ndarray,
                  rho_tot: float | np.ndarray) -> float | np.ndarray:
        """Contribución al tiempo relativo.

        δ_lat(S) = β · κ_lat(S) · ρ_lat(S) / ρ_tot(S)

        Args:
            S: Variable entrópica
            rho_tot: Densidad total en unidades de ρ_c

        Returns:
            Contribución al tiempo relativo
        """
        if not self.params.enabled:
            return np.zeros_like(S) if isinstance(S, np.ndarray) else 0.0

        kappa = self.kappa_lat(S)
        rho = self.rho_lat(S)
        rho_tot_safe = np.maximum(np.asarray(rho_tot), 1e-30)

        return self.params.beta * kappa * rho / rho_tot_safe

    def drho_lat_dS(self, S: float | np.ndarray) -> float | np.ndarray:
        """Derivada de ρ_lat respecto a S.

        dρ_lat/dS = -κ_lat(S) · ρ_lat(S)

        Args:
            S: Variable entrópica

        Returns:
            Derivada dρ_lat/dS
        """
        return -self.kappa_lat(S) * self.rho_lat(S)

    def w_lat(self, S: float | np.ndarray,
              dS_dz: float | np.ndarray) -> float | np.ndarray:
        """Índice de ecuación de estado efectivo.

        w_lat(z) = -1 + Δw_lat(z)
        Δw_lat ≃ (1/3) d ln ρ_lat / d ln(1+z)

        Para decaimiento entrópico puro, w_lat → -1 (comportamiento DE-like)
        pero con pequeñas correcciones temporales.

        Args:
            S: Variable entrópica
            dS_dz: Derivada dS/dz (de la Ley de Cronos)

        Returns:
            Ecuación de estado efectiva
        """
        if not self.params.enabled:
            return -np.ones_like(S) if isinstance(S, np.ndarray) else -1.0

        # d ln ρ_lat / dS = -κ_lat
        dlnrho_dS = -self.kappa_lat(S)

        # d ln(1+z) / dS ≈ -dS_dz^{-1} / (1+z)
        # Para simplificar, aproximamos Δw_lat
        dS_dz_safe = np.maximum(np.abs(np.asarray(dS_dz)), 1e-12)

        delta_w = (1.0 / 3.0) * dlnrho_dS * dS_dz_safe

        return -1.0 + delta_w

    def f_lat(self, S: float | np.ndarray,
              rho_crit: float = 1.0) -> float | np.ndarray:
        """Fracción de densidad latente respecto a ρ_crit.

        f_lat(S) ≡ ρ_lat(S) / ρ_crit

        El CMB impone f_lat(z_★) ≲ O(10⁻²).

        Args:
            S: Variable entrópica
            rho_crit: Densidad crítica (por defecto normalizada a 1)

        Returns:
            Fracción f_lat
        """
        return self.rho_lat(S) / rho_crit


def rho_lat_of_z(z: np.ndarray, params: LatentChannelParams,
                 S_of_z: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Calcula ρ_lat(z) dado un mapeo S(z).

    Wrapper conveniente para usar con código existente en z.

    Args:
        z: Array de redshifts
        params: Parámetros del canal latente
        S_of_z: Función que mapea z → S

    Returns:
        Array de ρ_lat(z)
    """
    channel = LatentChannel(params)
    S = S_of_z(z)
    return channel.rho_lat_array(S)


def hubble_correction_lat(z: np.ndarray, H_LCDM: np.ndarray,
                          rho_lat: np.ndarray,
                          delta_rho_id: np.ndarray | None = None) -> np.ndarray:
    """Corrección al Hubble por canal latente.

    H²_MCMC(z) / H²_ΛCDM(z) ≈ 1 + [ρ_lat(z) + Δρ_id(z)] / ρ_ΛCDM(z)

    Args:
        z: Array de redshifts
        H_LCDM: H(z) de ΛCDM en km/s/Mpc
        rho_lat: Densidad latente en unidades de ρ_c
        delta_rho_id: Corrección ρ_id - ρ_Λ (opcional)

    Returns:
        H_MCMC corregido
    """
    # Corrección total
    correction = rho_lat.copy()
    if delta_rho_id is not None:
        correction += delta_rho_id

    # Factor de corrección al H²
    # Asumiendo rho_lat está en unidades de ρ_c
    factor = 1.0 + correction

    return H_LCDM * np.sqrt(np.maximum(factor, 0.0))
