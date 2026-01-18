from __future__ import annotations

from typing import List


def build_H_eff(a: List[float], alpha_eff: List[float], H0: float) -> List[float]:
    if len(a) != len(alpha_eff):
        raise ValueError("a y alpha_eff deben tener la misma longitud.")
    H: List[float] = []
    for ai, al in zip(a, alpha_eff):
        H.append(H0 * (1.0 / max(ai, 1e-12)) ** al)
    return H
