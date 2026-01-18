"""
Módulo de Verificaciones Ontológicas
====================================

Este módulo implementa las pruebas de coherencia ontológica del MCMC
que deben ejecutarse como precondición para inferencia o N-body.

Verificaciones incluidas (Bloque 1):
1. Sellos: S1, S2, S3, S4 exactamente en la malla
2. Normalización: a(S4)=1, t_rel(S4)=0, H(S4)=H0
3. Monotonía y positividad: a>0, H>0, a creciente
4. Estabilidad de fluido oscuro (si perturbaciones activas)
5. Fracción latente en recombinación (si ρ_lat activo)

Referencias:
    - Tratado MCMC Maestro: validación ontológica del núcleo
    - Apartado Computacional: tests automáticos
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .s_grid import Seals, SGrid


@dataclass
class CheckResult:
    """Resultado de una verificación individual."""
    name: str
    passed: bool
    message: str
    details: Optional[Dict] = None


@dataclass
class ValidationReport:
    """Reporte completo de validación ontológica."""
    checks: List[CheckResult]

    @property
    def all_passed(self) -> bool:
        """True si todas las verificaciones pasaron."""
        return all(c.passed for c in self.checks)

    @property
    def n_passed(self) -> int:
        """Número de verificaciones que pasaron."""
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_failed(self) -> int:
        """Número de verificaciones que fallaron."""
        return sum(1 for c in self.checks if not c.passed)

    def __str__(self) -> str:
        lines = ["=" * 60]
        lines.append("REPORTE DE VALIDACIÓN ONTOLÓGICA MCMC")
        lines.append("=" * 60)

        for check in self.checks:
            status = "✓ PASS" if check.passed else "✗ FAIL"
            lines.append(f"\n[{status}] {check.name}")
            lines.append(f"    {check.message}")

        lines.append("\n" + "-" * 60)
        lines.append(f"RESUMEN: {self.n_passed}/{len(self.checks)} verificaciones exitosas")

        if not self.all_passed:
            lines.append("⚠ ATENCIÓN: Hay verificaciones fallidas")

        return "\n".join(lines)

    def raise_on_failure(self):
        """Lanza excepción si alguna verificación falló."""
        if not self.all_passed:
            failed = [c for c in self.checks if not c.passed]
            msg = "Validación ontológica fallida:\n"
            msg += "\n".join(f"  - {c.name}: {c.message}" for c in failed)
            raise ValueError(msg)


def check_seals_on_grid(
    S: np.ndarray,
    seals: Seals,
    tol: float = 1e-12
) -> CheckResult:
    """
    Verifica que todos los sellos están exactamente en la rejilla.

    Args:
        S: Array con la rejilla entrópica
        seals: Objeto Seals con los nodos críticos
        tol: Tolerancia para coincidencia exacta

    Returns:
        CheckResult con el resultado de la verificación
    """
    seal_dict = seals.get_all()
    missing = []

    for name, val in seal_dict.items():
        if not np.any(np.isclose(S, val, atol=tol)):
            missing.append(f"{name}={val}")

    if missing:
        return CheckResult(
            name="Sellos en rejilla",
            passed=False,
            message=f"Sellos faltantes: {', '.join(missing)}",
            details={"missing_seals": missing}
        )
    else:
        return CheckResult(
            name="Sellos en rejilla",
            passed=True,
            message="Todos los sellos (S1, S2, S3, S4) están en la rejilla"
        )


def check_normalization(
    S: np.ndarray,
    a: np.ndarray,
    H: np.ndarray,
    H0: float,
    t_rel: Optional[np.ndarray] = None,
    tol_a: float = 1e-10,
    tol_H: float = 1e-6,
    tol_t: float = 1e-10
) -> CheckResult:
    """
    Verifica normalización en S4 (último punto).

    Condiciones:
    - a(S4) = 1
    - H(S4) = H0
    - t_rel(S4) = 0 (si se proporciona)

    Args:
        S: Rejilla entrópica
        a: Factor de escala
        H: Parámetro de Hubble
        H0: Constante de Hubble esperada
        t_rel: Tiempo relativo (opcional)
        tol_a, tol_H, tol_t: Tolerancias

    Returns:
        CheckResult con el resultado
    """
    errors = []

    # a(S4) = 1
    if not np.isclose(a[-1], 1.0, atol=tol_a):
        errors.append(f"a(S4) = {a[-1]:.10f} ≠ 1")

    # H(S4) = H0
    if not np.isclose(H[-1], H0, atol=tol_H):
        errors.append(f"H(S4) = {H[-1]:.6f} ≠ H0 = {H0}")

    # t_rel(S4) = 0
    if t_rel is not None:
        if not np.isclose(t_rel[-1], 0.0, atol=tol_t):
            errors.append(f"t_rel(S4) = {t_rel[-1]:.10f} ≠ 0")

    if errors:
        return CheckResult(
            name="Normalización en S4",
            passed=False,
            message="; ".join(errors),
            details={"a_S4": a[-1], "H_S4": H[-1], "H0": H0}
        )
    else:
        return CheckResult(
            name="Normalización en S4",
            passed=True,
            message=f"a(S4)=1, H(S4)=H0={H0} verificados"
        )


def check_monotonicity(
    a: np.ndarray,
    S: np.ndarray = None
) -> CheckResult:
    """
    Verifica que a(S) es estrictamente creciente.

    Args:
        a: Factor de escala
        S: Rejilla entrópica (opcional, para reporte detallado)

    Returns:
        CheckResult con el resultado
    """
    da = np.diff(a)
    violations = np.sum(da <= 0)

    if violations > 0:
        # Encontrar dónde ocurren las violaciones
        violation_idx = np.where(da <= 0)[0]
        details = {
            "n_violations": violations,
            "violation_indices": violation_idx.tolist()
        }
        if S is not None:
            details["violation_S_values"] = S[violation_idx].tolist()

        return CheckResult(
            name="Monotonía de a(S)",
            passed=False,
            message=f"{violations} puntos con da/dS ≤ 0",
            details=details
        )
    else:
        return CheckResult(
            name="Monotonía de a(S)",
            passed=True,
            message="a(S) es estrictamente creciente en todo el rango"
        )


def check_positivity(
    a: np.ndarray,
    H: np.ndarray
) -> CheckResult:
    """
    Verifica positividad: a > 0 y H > 0.

    Args:
        a: Factor de escala
        H: Parámetro de Hubble

    Returns:
        CheckResult con el resultado
    """
    errors = []

    n_a_negative = np.sum(a <= 0)
    n_H_negative = np.sum(H <= 0)

    if n_a_negative > 0:
        errors.append(f"a ≤ 0 en {n_a_negative} puntos")
    if n_H_negative > 0:
        errors.append(f"H ≤ 0 en {n_H_negative} puntos")

    if errors:
        return CheckResult(
            name="Positividad",
            passed=False,
            message="; ".join(errors),
            details={
                "a_min": float(np.min(a)),
                "H_min": float(np.min(H))
            }
        )
    else:
        return CheckResult(
            name="Positividad",
            passed=True,
            message=f"a > 0 (min={np.min(a):.6e}), H > 0 (min={np.min(H):.2f})"
        )


def check_fluid_stability(
    cs2: np.ndarray,
    z: np.ndarray = None,
    cs2_min: float = 0.0,
    cs2_max: float = 1.0
) -> CheckResult:
    """
    Verifica estabilidad del fluido oscuro: 0 ≤ c_s^2 ≤ 1.

    Para estabilidad se requiere que la velocidad del sonido al cuadrado
    esté acotada. En el MCMC se recomienda c_s^2 ≈ 1 para evitar
    inestabilidades en perturbaciones.

    Args:
        cs2: Velocidad del sonido al cuadrado del fluido
        z: Redshift (opcional, para reporte)
        cs2_min: Cota inferior aceptable
        cs2_max: Cota superior aceptable

    Returns:
        CheckResult con el resultado
    """
    violations_low = np.sum(cs2 < cs2_min)
    violations_high = np.sum(cs2 > cs2_max)

    if violations_low > 0 or violations_high > 0:
        return CheckResult(
            name="Estabilidad de fluido oscuro",
            passed=False,
            message=f"c_s^2 fuera de [{cs2_min}, {cs2_max}]: "
                    f"{violations_low} bajo, {violations_high} alto",
            details={
                "cs2_min_actual": float(np.min(cs2)),
                "cs2_max_actual": float(np.max(cs2))
            }
        )
    else:
        return CheckResult(
            name="Estabilidad de fluido oscuro",
            passed=True,
            message=f"c_s^2 ∈ [{np.min(cs2):.4f}, {np.max(cs2):.4f}]"
        )


def check_early_dark_energy_fraction(
    Omega_de: np.ndarray,
    z: np.ndarray,
    z_recomb: float = 1089.0,
    max_fraction: float = 0.05
) -> CheckResult:
    """
    Verifica que la fracción de energía oscura sea pequeña en recombinación.

    Restricción importante del CMB: early dark energy no puede ser dominante.

    Args:
        Omega_de: Fracción de densidad de energía oscura
        z: Array de redshift
        z_recomb: Redshift de recombinación
        max_fraction: Fracción máxima permitida en recomb

    Returns:
        CheckResult con el resultado
    """
    # Encontrar el punto más cercano a recombinación
    idx_recomb = np.argmin(np.abs(z - z_recomb))
    Omega_de_recomb = Omega_de[idx_recomb]

    if Omega_de_recomb > max_fraction:
        return CheckResult(
            name="Early dark energy",
            passed=False,
            message=f"Ω_DE(z={z_recomb}) = {Omega_de_recomb:.4f} > {max_fraction}",
            details={
                "Omega_de_recomb": float(Omega_de_recomb),
                "z_recomb": z_recomb,
                "max_allowed": max_fraction
            }
        )
    else:
        return CheckResult(
            name="Early dark energy",
            passed=True,
            message=f"Ω_DE(z={z_recomb}) = {Omega_de_recomb:.4f} < {max_fraction}"
        )


def validate_background_solution(
    S: np.ndarray,
    a: np.ndarray,
    H: np.ndarray,
    H0: float,
    seals: Seals,
    t_rel: Optional[np.ndarray] = None
) -> ValidationReport:
    """
    Ejecuta todas las verificaciones ontológicas del fondo.

    Esta función debe llamarse como precondición antes de:
    - Ejecutar inferencia bayesiana
    - Generar condiciones iniciales para N-body
    - Acoplar con CLASS/CAMB

    Args:
        S: Rejilla entrópica
        a: Factor de escala
        H: Parámetro de Hubble
        H0: Constante de Hubble
        seals: Sellos ontológicos
        t_rel: Tiempo relativo (opcional)

    Returns:
        ValidationReport con todos los resultados
    """
    checks = [
        check_seals_on_grid(S, seals),
        check_normalization(S, a, H, H0, t_rel),
        check_monotonicity(a, S),
        check_positivity(a, H),
    ]

    return ValidationReport(checks=checks)


def quick_validate(
    S: np.ndarray,
    a: np.ndarray,
    H: np.ndarray,
    H0: float,
    seals: Seals = None
) -> bool:
    """
    Validación rápida que retorna True/False sin detalles.

    Útil para verificaciones en bucles de inferencia.

    Args:
        S, a, H, H0: Cantidades de fondo
        seals: Sellos (usa defaults si None)

    Returns:
        True si todas las verificaciones pasan
    """
    if seals is None:
        seals = Seals()

    report = validate_background_solution(S, a, H, H0, seals)
    return report.all_passed


# Exportaciones
__all__ = [
    'CheckResult',
    'ValidationReport',
    'check_seals_on_grid',
    'check_normalization',
    'check_monotonicity',
    'check_positivity',
    'check_fluid_stability',
    'check_early_dark_energy_fraction',
    'validate_background_solution',
    'quick_validate',
]
