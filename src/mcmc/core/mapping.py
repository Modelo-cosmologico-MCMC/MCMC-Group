from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def make_interpolators(z: np.ndarray, y: np.ndarray):
    """
    Interpolador y(z) estable. Se asume z monotonico creciente.
    """
    if not np.all(np.diff(z) > 0):
        # ordena por seguridad
        idx = np.argsort(z)
        z = z[idx]
        y = y[idx]
    return interp1d(z, y, kind="cubic", bounds_error=False, fill_value="extrapolate")


def S_to_z(a: np.ndarray) -> np.ndarray:
    return 1.0 / a - 1.0
