from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List


@dataclass(frozen=True)
class Block2Params:
    Mp0: float = 0.99
    Mpeq: float = 0.50
    gamma_S: float = 1.0
    kappa: float = 0.02
    alpha_base: float = 2.0 / 3.0
    delta_alpha: float = 0.20
    w0: float = -1.0
    delta_w: float = 0.02
    S1: float = 0.010
    S4: float = 1.001


def run_block2(S: List[float], phi_of_S: List[float] | None, p: Block2Params) -> Dict[str, object]:
    if phi_of_S is None:
        phi_of_S = [0.0 for _ in S]
    if len(phi_of_S) != len(S):
        raise ValueError("phi_of_S debe tener la misma longitud que S.")

    Mp_base: List[float] = []
    Ep_base: List[float] = []
    Mp_eff: List[float] = []
    Ep_eff: List[float] = []
    alpha_eff: List[float] = []
    w_eff: List[float] = []

    denom = max((p.S4 - p.S1), 1e-12)

    for Si, ph in zip(S, phi_of_S):
        x = max(0.0, min(1.0, (Si - p.S1) / denom))
        f = x ** p.gamma_S

        mpb = p.Mp0 - (p.Mp0 - p.Mpeq) * f
        epb = 1.0 - mpb

        dmp = p.kappa * math.tanh(ph)
        mpe = mpb + dmp
        epe = epb - dmp

        # clamp + renorm para unidad dual
        mpe = max(0.0, min(1.0, mpe))
        epe = max(0.0, min(1.0, epe))
        s = max(mpe + epe, 1e-12)
        mpe, epe = mpe / s, epe / s

        ratio = (epe - mpe) / max((epe + mpe), 1e-12)
        ae = p.alpha_base + p.delta_alpha * ratio
        we = p.w0 + p.delta_w * math.sin(ph)

        Mp_base.append(mpb)
        Ep_base.append(epb)
        Mp_eff.append(mpe)
        Ep_eff.append(epe)
        alpha_eff.append(ae)
        w_eff.append(we)

    return {
        "Mp_base": Mp_base,
        "Ep_base": Ep_base,
        "Mp_eff": Mp_eff,
        "Ep_eff": Ep_eff,
        "alpha_eff": alpha_eff,
        "w_eff": w_eff,
    }
