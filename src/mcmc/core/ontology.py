"""Ontological constants and thresholds for MCMC model.

CORRECCIÓN ONTOLÓGICA (2025):
El parámetro entrópico S del MCMC tiene rango [0, 100], NO [1.001, 1.0015].

RÉGIMEN PRE-GEOMÉTRICO: S ∈ [0, 1.001)
- S = 0: Estado primordial de máxima superposición (toda masa sin forma)
- Transiciones canónicas pre-geométricas preservadas
- No existe espacio-tiempo clásico

RÉGIMEN GEOMÉTRICO (POST-BIG BANG): S ∈ [1.001, 95.07]
- S_GEOM = 1.001: Big Bang - surge espacio-tiempo clásico
- S_actual ≈ 95.07: Presente cosmológico (calibrado con Ω_b = 0.0493)
- S → 100: Límite asintótico de Sitter (Ep domina completamente)

Correspondencia con ΛCDM:
- Masa determinada (Ω_b) = 4.93% → (100 - S_actual) ≈ 4.93%
- MCV/Materia oscura (Ω_DM) = 26.6% → manifestación emergente
- Ep/Energía oscura (Ω_Λ) = 68.5% → masa primordial convertida en espacio

El presente es estratificado: S_local(x) varía según densidad tensorial.
Las islas tensoriales (BH, cúmulos) tienen S_local < S_global.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ==============================================================================
# CONSTANTES FÍSICAS
# ==============================================================================
c = 299792458.0                          # m/s (velocidad de la luz)
G = 6.67430e-11                          # m³/(kg·s²) (constante gravitacional)
h_bar = 1.054571817e-34                  # J·s (constante de Planck reducida)

Mpc_to_m = 3.085677581e22
km_s_Mpc_to_Hz = 1e3 / Mpc_to_m


# ==============================================================================
# PARÁMETROS COSMOLÓGICOS (Planck 2018)
# ==============================================================================
OMEGA_B = 0.0493                # Materia bariónica (masa determinada)
OMEGA_DM = 0.266                # Materia oscura (MCV)
OMEGA_M = 0.315                 # Materia total
OMEGA_LAMBDA = 0.685            # Energía oscura (Ep)
H_0 = 67.4                      # km/s/Mpc
H_0_SI = H_0 * km_s_Mpc_to_Hz   # Hz

RHO_CRIT = 3 * H_0_SI**2 / (8 * np.pi * G)  # ~8.5e-27 kg/m³


# ==============================================================================
# PARÁMETROS ENTRÓPICOS DEL MCMC
# ==============================================================================
S_MIN = 0.0                     # Estado primordial (máxima superposición)
S_MAX = 100.0                   # Límite asintótico de Sitter
S_GEOM = 1.001                  # Big Bang: transición pre-geométrica → geométrica
S_0 = S_MAX * (1 - OMEGA_B)     # ≈ 95.07 (presente cosmológico)


@dataclass(frozen=True)
class OntologicalThresholds:
    """Umbrales ontológicos del MCMC (valores críticos de S).

    CORRECCIÓN: S ∈ [0, 100]
    - Pre-geométrico: S ∈ [0, 1.001)
    - Post-Big Bang: S ∈ [1.001, 95.07]

    Attributes:
        S_MIN: Estado primordial (máxima superposición)
        S_GEOM: Big Bang (transición pre-geométrica → espacio-tiempo clásico)
        S_RECOMB: Recombinación (z ≈ 1100)
        S_STAR_PEAK: Pico de formación estelar (z ≈ 2)
        S_0: Presente cosmológico (calibrado con Ω_b)
        S_MAX: Límite asintótico de Sitter
    """
    # Rango fundamental
    S_MIN: float = 0.0
    S_MAX: float = 100.0

    # Big Bang: transición pre-geométrica → espacio-tiempo clásico
    S_GEOM: float = 1.001

    # Transiciones pre-geométricas canónicas (S < S_GEOM)
    S_PRE_0: float = 0.001      # Primera singularidad pre-geom
    S_PRE_1: float = 0.01       # Segunda transición pre-geom
    S_PRE_2: float = 0.1        # Tercera transición pre-geom
    S_PRE_3: float = 0.5        # Cuarta transición pre-geom

    # === COMPATIBILIDAD: Aliases para código legacy ===
    # Estas constantes mapean los nombres antiguos a los nuevos
    S_PLANCK: float = 0.001     # Alias para S_PRE_0 (Planck scale)
    S_GUT: float = 0.01         # Alias para S_PRE_1 (GUT scale)
    S_EW: float = 0.5           # Alias para S_PRE_3 (Electroweak scale)
    S_BB: float = 1.001         # Alias para S_GEOM (Big Bang)

    # Épocas cosmológicas post-Big Bang (S ≥ S_GEOM)
    S_RECOMB: float = 1.08      # z ≈ 1100 (recombinación)
    S_GALAXY: float = 2.5       # z ≈ 10 (primeras galaxias)
    S_STAR_PEAK: float = 47.5   # z ≈ 2 (pico formación estelar)
    S_Z1: float = 65.0          # z ≈ 1 (SNe Ia referencia)
    S_Z05: float = 84.2         # z ≈ 0.5

    # Presente cosmológico
    S_0: float = 95.07          # z = 0 (calibrado con Ω_b = 0.0493)


@dataclass(frozen=True)
class OntologicalRegimes:
    """Define los regímenes principales del MCMC.

    Régimen pre-geométrico: S ∈ [0, 1.001)
        - No existe geometría clásica
        - Masa primordial superpuesta
        - Transiciones canónicas preservadas (S_PRE_0, S_PRE_1, S_PRE_2, S_PRE_3)

    Régimen geométrico (Post-Big Bang): S ∈ [1.001, 100]
        - Espacio-tiempo clásico
        - Conversión progresiva Mp → Ep
        - Cosmología observable
    """
    # Pre-geométrico: S ∈ [0, 1.001)
    PRE_GEOM_MIN: float = 0.0
    PRE_GEOM_MAX: float = 1.001

    # Geométrico (cosmología observable post-Big Bang): S ∈ [1.001, 100]
    GEOM_MIN: float = 1.001
    GEOM_MAX: float = 100.0


# Instancias canónicas
THRESHOLDS = OntologicalThresholds()
REGIMES = OntologicalRegimes()

# Paso de discretización
DS_CANONICAL = 0.01  # Mayor que antes dado el rango [0, 100]


# ==============================================================================
# FUNCIONES DE RÉGIMEN
# ==============================================================================

def is_pre_geometric(S: float) -> bool:
    """Verifica si S está en régimen pre-geométrico."""
    return S < THRESHOLDS.S_GEOM


def is_geometric(S: float) -> bool:
    """Verifica si S está en régimen geométrico (cosmología observable)."""
    return S >= THRESHOLDS.S_GEOM


def validate_S(S: float | np.ndarray) -> bool:
    """Valida que S esté en el rango permitido [0, 100]."""
    S_arr = np.asarray(S)
    return bool(np.all((S_arr >= S_MIN) & (S_arr <= S_MAX)))


def get_epoch_name(S: float) -> str:
    """Retorna el nombre de la época cosmológica para un valor de S.

    Pre-geométrico: S ∈ [0, 1.001)
    Post-Big Bang: S ∈ [1.001, 95.07]
    """
    if S < THRESHOLDS.S_PRE_1:
        return "pre-geométrico/primordial"
    elif S < THRESHOLDS.S_PRE_2:
        return "pre-geométrico/transición-1"
    elif S < THRESHOLDS.S_PRE_3:
        return "pre-geométrico/transición-2"
    elif S < THRESHOLDS.S_GEOM:
        return "pre-geométrico/pre-Big-Bang"
    elif S < THRESHOLDS.S_RECOMB:
        return "inflación/recalentamiento"
    elif S < THRESHOLDS.S_GALAXY:
        return "edad oscura"
    elif S < THRESHOLDS.S_STAR_PEAK:
        return "formación de estructuras"
    elif S < THRESHOLDS.S_Z1:
        return "pico formación estelar"
    elif S < THRESHOLDS.S_0:
        return "era de energía oscura"
    else:
        return "presente/futuro"


# ==============================================================================
# CONVERSIÓN MCMC ↔ ΛCDM
# ==============================================================================

def LCDM_to_MCMC(
    Omega_b: float = OMEGA_B,
    Omega_m: float = OMEGA_M,
    Omega_Lambda: float = OMEGA_LAMBDA
) -> dict:
    """Traduce parámetros ΛCDM al marco MCMC.

    Correspondencia:
    - Ω_b → masa determinada (único "resto" no convertido)
    - Ω_DM → MCV (manifestación emergente)
    - Ω_Λ → Ep (espacio/energía de conversión)

    Returns:
        dict con S_actual, masa_determinada, MCV, Ep, f_conversion
    """
    return {
        'S_actual': S_MAX * (1 - Omega_b),
        'masa_determinada': Omega_b,
        'MCV': Omega_m - Omega_b,
        'Ep': Omega_Lambda,
        'f_conversion': 1 - Omega_b,
    }


def MCMC_to_LCDM(
    S: float,
    masa_det: float,
    MCV: float,
    Ep: float
) -> dict:
    """Traduce parámetros MCMC a ΛCDM estándar."""
    S_esperado = S_MAX * (1 - masa_det)
    return {
        'Omega_b': masa_det,
        'Omega_DM': MCV,
        'Omega_m': masa_det + MCV,
        'Omega_Lambda': Ep,
        'S_check': S_esperado,
        'consistencia': abs(S - S_esperado) < 1.0
    }


# ==============================================================================
# COMPATIBILIDAD CON CÓDIGO ANTERIOR
# ==============================================================================
# Alias para compatibilidad con código que usaba los nombres antiguos
# DEPRECATED: Usar THRESHOLDS.S_GEOM en lugar de S_BB

def is_pre_bb(S: float) -> bool:
    """DEPRECATED: Usar is_pre_geometric(). Mantiene por compatibilidad."""
    return is_pre_geometric(S)


def is_post_bb(S: float) -> bool:
    """DEPRECATED: Usar is_geometric(). Mantiene por compatibilidad."""
    return is_geometric(S)
