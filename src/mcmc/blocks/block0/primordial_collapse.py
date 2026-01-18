from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .primordial_state import PrimordialState
from .pregeometric_field import PreGeometricField, PreGeometricFieldParams


@dataclass(frozen=True)
class Block0Params:
    Mp0: float = 0.999
    Ep0: float = 0.001
    eps: float = 0.001
    S_min: float = 0.001
    S_max: float = 0.009
    dS: float = 1e-3
    field_params: PreGeometricFieldParams = field(default_factory=PreGeometricFieldParams)


def run_block0(params: Block0Params) -> Dict[str, object]:
    """Cadena Σn: S∈[0.001,0.009] con ΔS=1e-3. Devuelve condiciones de contorno."""
    state = PrimordialState(Mp0=params.Mp0, Ep0=params.Ep0, eps=params.eps)
    field_obj = PreGeometricField(params.field_params)

    n_steps = int(round((params.S_max - params.S_min) / params.dS)) + 1
    S_nodes: List[float] = [params.S_min + i * params.dS for i in range(n_steps)]

    Mp: List[float] = [state.Mp0]
    Ep: List[float] = [state.Ep0]
    Phi: List[float] = []
    kpre: List[float] = []

    for S in S_nodes:
        phi = field_obj.phi(S)
        k = field_obj.collapse_rate(S)
        dMp = -k * params.dS
        dEp = +k * params.dS

        Mp.append(Mp[-1] + dMp)
        Ep.append(Ep[-1] + dEp)
        Phi.append(phi)
        kpre.append(k)

    # recortar al final físico [0,1] por robustez
    Mp_pre = max(0.0, min(1.0, Mp[-1]))
    Ep_pre = max(0.0, min(1.0, Ep[-1]))
    # renormalizar para preservar unidad dual
    s = Mp_pre + Ep_pre
    Mp_pre, Ep_pre = Mp_pre / s, Ep_pre / s

    return {
        "S_nodes": S_nodes,
        "Mp_chain": Mp,
        "Ep_chain": Ep,
        "phi_chain": Phi,
        "kpre_chain": kpre,
        "Mp_pre": Mp_pre,
        "Ep_pre": Ep_pre,
        "phi_pre": Phi[-1] if Phi else 0.0,
        "kpre_mean": sum(kpre) / max(len(kpre), 1),
    }
