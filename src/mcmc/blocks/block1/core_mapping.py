from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import math

from .cronos import CronosModel, CronosParams


@dataclass(frozen=True)
class Block1Params:
    S_min: float = 0.010
    S_max: float = 1.001
    dS: float = 1e-3
    H0: float = 67.4  # km/s/Mpc (normalización)
    # signo: + integra hacia el "presente" desde S_min a S_max
    sign: float = +1.0
    cronos: CronosParams = field(default_factory=CronosParams)


def run_block1(params: Block1Params) -> Dict[str, object]:
    model = CronosModel(params.cronos)
    n_steps = int(round((params.S_max - params.S_min) / params.dS)) + 1
    S: List[float] = [params.S_min + i * params.dS for i in range(n_steps)]

    # integrar ln a y t_rel
    ln_a: List[float] = [0.0]
    t_rel: List[float] = [0.0]

    for i in range(1, len(S)):
        Si = S[i - 1]
        dln_a = params.sign * model.C(Si) * params.dS
        dt = params.sign * model.T(Si) * model.N(Si) * params.dS
        ln_a.append(ln_a[-1] + dln_a)
        t_rel.append(t_rel[-1] + dt)

    # normalizar a(S4)=1: convertir ln_a a a y re-escalar
    a_raw = [math.exp(x) for x in ln_a]
    a_norm = a_raw[-1]
    a = [x / a_norm for x in a_raw]

    # z = 1/a - 1 (para a<=1)
    z = [(1.0 / max(ai, 1e-12)) - 1.0 for ai in a]

    # H(S): MVP -> usar H0 * (a(S4)/a(S))^{1} como baseline (Bloque2 refina)
    H = [params.H0 * (1.0 / max(ai, 1e-12)) for ai in a]

    return {
        "S": S,
        "a": a,
        "z": z,
        "t_rel": t_rel,
        "H": H,
        "H0": params.H0,
    }
