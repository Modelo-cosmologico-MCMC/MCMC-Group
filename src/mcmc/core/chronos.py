"""Chronos Law: S↔t mapping with Big Bang anchor.

The Chronos law defines the relationship between entropic variable S
and cosmic time t. Time is anchored at the Big Bang:

    t(S_BB) = 0

where S_BB = 1.001 is the Big Bang observable threshold.

Core equations (from contract v2.0):
    dt_rel/dS = T(S) * N(S)
    t_rel(S) = integral[S_PLANCK, S] T(s) * N(s) ds

where:
    T(S) = chronification function (peaks at thresholds)
    N(S) = exp(Phi_ten(S)) = lapse function
    Phi_ten(S) = tensional field
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mcmc.core.ontology import THRESHOLDS
from mcmc.core.cronoshapes import CronosShapeParams, T_of_S, N_of_S


@dataclass(frozen=True)
class ChronosParams:
    """Parameters for the Chronos law t_rel(S) anchored at S_BB.

    Uses CronosShapeParams internally for T(S) and N(S) evaluation.
    """
    # Shape parameters for T(S), Phi_ten(S), N(S)
    shapes: CronosShapeParams = CronosShapeParams()

    # Thresholds (can override defaults)
    S_PLANCK: float = THRESHOLDS.S_PLANCK
    S_GUT: float = THRESHOLDS.S_GUT
    S_EW: float = THRESHOLDS.S_EW
    S_BB: float = THRESHOLDS.S_BB


def _as_array(x) -> np.ndarray:
    """Convert input to float array."""
    return np.atleast_1d(np.asarray(x, dtype=float))


def dt_dS(S: np.ndarray, p: ChronosParams) -> np.ndarray:
    """Compute dt/dS = T(S) * N(S) according to Chronos law.

    Args:
        S: Array of entropy values
        p: Chronos parameters

    Returns:
        Array of dt/dS values
    """
    S_arr = _as_array(S)
    T = T_of_S(S_arr, p.shapes, S1=p.S_PLANCK, S2=p.S_GUT, S3=p.S_EW)
    N = N_of_S(S_arr, p.shapes, S1=p.S_PLANCK, S2=p.S_GUT, S3=p.S_EW)
    return T * N


def t_rel_raw(S: np.ndarray, p: ChronosParams, S_start: float | None = None) -> np.ndarray:
    """Compute raw relative time t_rel(S) by integrating T(S)*N(S).

    Integrates from S_start to each value in S.
    NOT anchored at BB (use t_of_S for anchored version).

    Args:
        S: Array of entropy values (must be sorted for efficiency)
        p: Chronos parameters
        S_start: Starting point for integration (default: S_PLANCK)

    Returns:
        Array of raw relative time values
    """
    S_arr = _as_array(S)
    S_start = p.S_PLANCK if S_start is None else float(S_start)

    # Build fine grid for integration
    S_min = min(S_start, float(np.min(S_arr)))
    S_max = float(np.max(S_arr))
    n_points = max(1000, int((S_max - S_min) / 1e-4))
    S_grid = np.linspace(S_min, S_max, n_points)

    # Evaluate integrand on grid
    integrand = dt_dS(S_grid, p)

    # Cumulative trapezoid integration
    dS = np.diff(S_grid)
    t_cumulative = np.zeros(len(S_grid))
    for i in range(1, len(S_grid)):
        t_cumulative[i] = t_cumulative[i - 1] + 0.5 * dS[i - 1] * (integrand[i] + integrand[i - 1])

    # Interpolate to requested S values
    from scipy.interpolate import interp1d
    interp = interp1d(S_grid, t_cumulative, kind="linear", bounds_error=False, fill_value="extrapolate")
    return interp(S_arr)


def t_of_S(S, p: ChronosParams, S_BB: float | None = None) -> np.ndarray:
    """Compute Chronos time ANCHORED at Big Bang: t(S_BB) = 0.

    This is the primary function for converting S to cosmic time.

    Args:
        S: Entropy value(s)
        p: Chronos parameters
        S_BB: Big Bang threshold (default: from THRESHOLDS)

    Returns:
        Time array with t(S_BB) = 0
        - t < 0 for S < S_BB (pre-Big-Bang)
        - t > 0 for S > S_BB (post-Big-Bang, observable universe)
    """
    S_BB = p.S_BB if S_BB is None else float(S_BB)
    S_arr = _as_array(S)

    # Compute raw time for input S and for S_BB
    t_raw = t_rel_raw(S_arr, p)
    t_bb = float(t_rel_raw(np.array([S_BB]), p)[0])

    # Anchor at BB
    return t_raw - t_bb


def _eval_t_scalar(S_val: float, p: ChronosParams) -> float:
    """Evaluate t_of_S for a single scalar value."""
    result = t_of_S(np.array([S_val]), p)
    return float(result[0])


def S_of_t(t, p: ChronosParams, bracket: tuple[float, float] | None = None, n_iter: int = 80) -> np.ndarray:
    """Invert Chronos: given t, return S such that t_of_S(S) = t.

    Uses robust bisection (no scipy.optimize dependency).

    Args:
        t: Time value(s) (t=0 at Big Bang)
        p: Chronos parameters
        bracket: (S_min, S_max) for search (auto-expanded if needed)
        n_iter: Number of bisection iterations

    Returns:
        Array of S values corresponding to input times
    """
    t_arr = _as_array(t)
    if bracket is None:
        bracket = (0.01, p.S_BB + 5.0)

    a, b = float(bracket[0]), float(bracket[1])
    out = np.empty_like(t_arr)

    for idx, ti in np.ndenumerate(t_arr):
        lo, hi = a, b
        f_lo = _eval_t_scalar(lo, p)
        f_hi = _eval_t_scalar(hi, p)

        # Expand hi if needed (t increasing with S)
        expand = 0
        while f_hi < ti and expand < 30:
            hi *= 2.0
            f_hi = _eval_t_scalar(hi, p)
            expand += 1

        # Expand lo if needed
        expand = 0
        while f_lo > ti and expand < 30:
            lo = max(lo * 0.5, 1e-6)
            f_lo = _eval_t_scalar(lo, p)
            expand += 1

        # Bisection
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            f_mid = _eval_t_scalar(mid, p)
            if f_mid < ti:
                lo = mid
            else:
                hi = mid

        out[idx] = 0.5 * (lo + hi)

    return out


def assert_monotonic(p: ChronosParams, S_range: tuple[float, float] = (0.001, 2.0)) -> None:
    """Assert that t(S) is monotonically increasing.

    Raises:
        AssertionError: If t(S) is not monotonic in the given range
    """
    S = np.linspace(S_range[0], S_range[1], 500)
    t = t_of_S(S, p)
    dt = np.diff(t)
    assert np.all(dt > 0), f"t(S) not monotonic: min(dt)={np.min(dt)}"
