"""Extracción del mass gap desde correladores gauge.

El mass gap se extrae de correladores en tiempo euclídeo:

    C(τ; S) = ⟨O(x⃗, τ) O†(0⃗, 0)⟩_S ∝ exp(-E_min(S) · τ)

Procedimiento:
    1. Generar N_conf configuraciones gauge tras termalización
    2. Calcular correladores C(τ; S_k) para separaciones temporales τ
    3. Ajustar la caída exponencial para extraer E_min(S_k)
    4. Convertir a unidades físicas mediante relaciones de escalado

Conexión ontológica:
    E_min(S = 1.001) ≃ α_H · E_ref
    donde E_ref es una escala de referencia (ΛQCD o escala GUT).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import curve_fit

from .wilson import WilsonAction, WilsonActionParams
from .monte_carlo import MetropolisSampler, MCParams
from .beta_of_S import beta_of_S, BetaParams


@dataclass
class MassGapResult:
    """Resultado de extracción de mass gap.

    Attributes:
        S: Valor de S
        beta: Valor de β(S)
        E_min: Mass gap extraído (unidades de retícula)
        E_min_err: Error en E_min
        chi2_dof: χ²/dof del ajuste
        correlator: Correlador promediado
        correlator_err: Error del correlador
    """
    S: float
    beta: float
    E_min: float
    E_min_err: float
    chi2_dof: float
    correlator: np.ndarray
    correlator_err: np.ndarray


class MassGapExtractor:
    """Extrae el mass gap de configuraciones gauge.

    Usa correladores de Wilson loops o plaquette-plaquette
    para medir la masa del estado más ligero.

    Attributes:
        L: Tamaño de retícula temporal
    """

    def __init__(self, L_t: int):
        """Inicializa el extractor.

        Args:
            L_t: Extensión temporal de la retícula
        """
        self.L_t = L_t

    def plaquette_correlator(
        self,
        configs: list[np.ndarray],
        params: WilsonActionParams
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calcula correlador plaquette-plaquette.

        C(τ) = ⟨P(0) P(τ)⟩ - ⟨P⟩²

        Args:
            configs: Lista de configuraciones (enlaces)
            params: Parámetros de la acción

        Returns:
            (tau, C_tau): Separaciones temporales y correlador
        """
        n_conf = len(configs)
        L = params.L
        n_dim = params.n_dim

        # Array para correladores por configuración
        C_all = np.zeros((n_conf, L))

        for ic, links in enumerate(configs):
            # Crear acción temporal para acceder a plaquettes
            action = WilsonAction(params)
            action.links = links

            # Plaquette en cada tiempo
            P_t = np.zeros(L)
            count_t = np.zeros(L)

            for t in range(L):
                for x in np.ndindex(*([L] * (n_dim - 1))):
                    site = (t,) + x
                    # Promedio sobre plaquettes espaciales
                    for mu in range(1, n_dim):
                        for nu in range(mu + 1, n_dim):
                            P_t[t] += action._plaquette(site, mu, nu)
                            count_t[t] += 1

            P_t /= np.maximum(count_t, 1)

            # Correlador conectado
            P_mean = np.mean(P_t)
            for tau in range(L):
                C_all[ic, tau] = np.mean(
                    (P_t - P_mean) * np.roll(P_t - P_mean, -tau)
                )

        # Promediar sobre configuraciones
        C_mean = np.mean(C_all, axis=0)
        C_err = np.std(C_all, axis=0) / np.sqrt(n_conf)

        tau = np.arange(L)
        return tau, C_mean, C_err

    def polyakov_correlator(
        self,
        configs: list[np.ndarray],
        params: WilsonActionParams,
        direction: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calcula correlador de líneas de Polyakov.

        P(x⃗) = Tr Π_{t=0}^{L_t-1} U_0(t, x⃗)

        C(r) = ⟨P(0) P†(r)⟩

        Relacionado con el potencial entre quarks estáticos.

        Args:
            configs: Lista de configuraciones
            params: Parámetros de la acción
            direction: Dirección espacial para la separación

        Returns:
            (r, C_r, C_err): Separación y correlador
        """
        n_conf = len(configs)
        L = params.L

        # Para U(1), la línea de Polyakov es exp(i Σ_t θ_t)
        C_all = np.zeros((n_conf, L))

        for ic, links in enumerate(configs):
            # Calcular líneas de Polyakov en una rebanada espacial
            P_line = np.zeros(L, dtype=complex)

            for x in range(L):
                # Producto de enlaces en dirección temporal
                phase = 0.0
                for t in range(L):
                    if direction == 0:
                        site = (t, x, 0, 0) if params.n_dim == 4 else (t, x, 0)
                    else:
                        site = (t, 0, x, 0) if params.n_dim == 4 else (t, 0, x)
                    phase += links[site][0]  # Enlace temporal

                P_line[x] = np.exp(1j * phase)

            # Correlador
            for r in range(L):
                C_all[ic, r] = np.real(
                    np.mean(np.conj(P_line) * np.roll(P_line, -r))
                )

        C_mean = np.mean(C_all, axis=0)
        C_err = np.std(C_all, axis=0) / np.sqrt(n_conf)

        r = np.arange(L)
        return r, C_mean, C_err

    @staticmethod
    def _exp_decay(tau: np.ndarray, A: float, m: float) -> np.ndarray:
        """Modelo de decaimiento exponencial.

        C(τ) = A · exp(-m·τ)
        """
        return A * np.exp(-m * tau)

    @staticmethod
    def _cosh_decay(tau: np.ndarray, A: float, m: float, L: int) -> np.ndarray:
        """Modelo cosh para condiciones periódicas.

        C(τ) = A · [exp(-m·τ) + exp(-m·(L-τ))]
        """
        return A * (np.exp(-m * tau) + np.exp(-m * (L - tau)))

    def extract_mass(
        self,
        tau: np.ndarray,
        C: np.ndarray,
        C_err: np.ndarray,
        use_cosh: bool = True,
        tau_min: int = 1,
        tau_max: int | None = None
    ) -> tuple[float, float, float]:
        """Extrae la masa del correlador.

        Ajusta C(τ) a una exponencial o cosh para obtener E_min.

        Args:
            tau: Separaciones temporales
            C: Correlador
            C_err: Error del correlador
            use_cosh: Si True, usa modelo cosh
            tau_min: τ mínimo para el ajuste
            tau_max: τ máximo para el ajuste

        Returns:
            (mass, mass_err, chi2_dof): Masa extraída, error y χ²/dof
        """
        L = len(tau)
        if tau_max is None:
            tau_max = L // 2

        # Rango de ajuste
        mask = (tau >= tau_min) & (tau <= tau_max)
        tau_fit = tau[mask]
        C_fit = C[mask]
        err_fit = np.maximum(C_err[mask], 1e-15)

        # Valores iniciales
        A_init = np.abs(C_fit[0])
        m_init = 0.5  # Guess inicial

        try:
            if use_cosh:
                def model(t, A, m):
                    return self._cosh_decay(t, A, m, L)
            else:
                model = self._exp_decay

            popt, pcov = curve_fit(
                model,
                tau_fit,
                C_fit,
                p0=[A_init, m_init],
                sigma=err_fit,
                absolute_sigma=True,
                bounds=([0, 0], [np.inf, 10.0])
            )

            mass = popt[1]
            mass_err = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else 0.1 * mass

            # χ²
            C_model = model(tau_fit, *popt)
            chi2 = np.sum(((C_fit - C_model) / err_fit)**2)
            dof = len(tau_fit) - 2
            chi2_dof = chi2 / max(dof, 1)

        except (RuntimeError, ValueError):
            # Ajuste falló: usar masa efectiva
            mass = self._effective_mass(C, tau_min)
            mass_err = 0.5 * mass
            chi2_dof = np.nan

        return mass, mass_err, chi2_dof

    def _effective_mass(self, C: np.ndarray, t: int = 1) -> float:
        """Calcula masa efectiva local.

        m_eff(t) = ln[C(t) / C(t+1)]
        """
        if t >= len(C) - 1:
            t = len(C) // 4

        C_t = np.abs(C[t])
        C_t1 = np.abs(C[t + 1])

        if C_t1 < 1e-15 or C_t < 1e-15:
            return 1.0

        return np.log(C_t / C_t1)


def extract_mass_gap(
    configs: list[np.ndarray],
    params: WilsonActionParams,
    S: float,
    beta: float
) -> MassGapResult:
    """Extrae mass gap de un conjunto de configuraciones.

    Args:
        configs: Configuraciones gauge
        params: Parámetros de la acción
        S: Valor de S
        beta: Valor de β

    Returns:
        Resultado con mass gap y diagnósticos
    """
    extractor = MassGapExtractor(params.L)

    # Calcular correlador
    tau, C, C_err = extractor.plaquette_correlator(configs, params)

    # Extraer masa
    mass, mass_err, chi2_dof = extractor.extract_mass(tau, C, C_err)

    return MassGapResult(
        S=S,
        beta=beta,
        E_min=mass,
        E_min_err=mass_err,
        chi2_dof=chi2_dof,
        correlator=C,
        correlator_err=C_err,
    )


def run_S_scan(
    S_values: np.ndarray,
    beta_params: BetaParams,
    lattice_params: WilsonActionParams,
    mc_params: MCParams,
    progress_callback: callable | None = None
) -> list[MassGapResult]:
    """Ejecuta barrido en S para extraer E_min(S).

    Args:
        S_values: Array de valores de S
        beta_params: Parámetros para β(S)
        lattice_params: Parámetros de retícula
        mc_params: Parámetros Monte Carlo
        progress_callback: Función de progreso (i, n_total)

    Returns:
        Lista de resultados MassGapResult
    """
    results = []
    n_total = len(S_values)

    for i, S in enumerate(S_values):
        # Calcular β(S)
        beta = float(beta_of_S(S, beta_params))

        # Crear acción con β actualizado
        params = WilsonActionParams(
            beta=beta,
            N_color=lattice_params.N_color,
            L=lattice_params.L,
            n_dim=lattice_params.n_dim,
        )
        action = WilsonAction(params)
        action.hot_start(seed=mc_params.seed + i)

        # Monte Carlo
        sampler = MetropolisSampler(action, mc_params)
        configs = sampler.generate_configurations()

        # Extraer mass gap
        result = extract_mass_gap(configs, params, S, beta)
        results.append(result)

        # Callback de progreso
        if progress_callback is not None:
            progress_callback(i + 1, n_total)

    return results


def mass_gap_to_physical(
    E_lat: float,
    a_lat: float,
    hbar_c: float = 197.3  # MeV·fm
) -> float:
    """Convierte mass gap de unidades de retícula a MeV.

    E_phys = E_lat / a_lat [en unidades de hbar·c/a]
           = E_lat · hbar·c / a_lat

    Args:
        E_lat: Mass gap en unidades de retícula
        a_lat: Espaciado de retícula en fm
        hbar_c: hbar·c en MeV·fm

    Returns:
        Mass gap en MeV
    """
    return E_lat * hbar_c / a_lat


def alpha_H_from_mass_gap(
    E_min_S4: float,
    E_ref: float = 200.0  # ΛQCD en MeV
) -> float:
    """Calcula α_H desde el mass gap en S₄.

    E_min(S₄) ≃ α_H · E_ref

    Args:
        E_min_S4: Mass gap en S = S₄ = 1.001 (MeV)
        E_ref: Escala de referencia (MeV)

    Returns:
        Parámetro α_H
    """
    return E_min_S4 / E_ref
