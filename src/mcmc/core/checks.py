from __future__ import annotations

import numpy as np


def assert_background_ok(sol: dict, *, atol: float = 1e-10) -> None:
    a = sol["a"]
    H = sol["H"]

    if not np.isfinite(a).all() or not np.isfinite(H).all():
        raise ValueError("NaNs/Infs en solucion de fondo.")

    if np.any(a <= 0):
        raise ValueError("a(S) debe ser > 0.")

    if np.any(H <= 0):
        raise ValueError("H(S) debe ser > 0.")

    if abs(a[-1] - 1.0) > atol:
        raise ValueError(f"Normalizacion rota: a(S4)={a[-1]} != 1.")

    if not np.all(np.diff(a) >= 0):
        # a debe crecer hacia S4 en este sentido de integracion
        raise ValueError("Monotonia rota: a(S) no crece hacia S4.")
