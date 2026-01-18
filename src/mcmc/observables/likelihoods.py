from __future__ import annotations

from .hz import chi2_hz
from .sne import chi2_sne
from .bao import chi2_bao


def loglike_total(datasets: dict, model: dict) -> float:
    """
    datasets keys: 'hz','sne','bao' (opcionales)
    model funcs: model['H(z)'], model['mu(z)'], model['DVrd(z)']
    """
    chi2 = 0.0

    if "hz" in datasets:
        d = datasets["hz"]
        chi2 += chi2_hz(d["z"], d["H"], d["sigma"], model["H(z)"])

    if "sne" in datasets:
        d = datasets["sne"]
        chi2 += chi2_sne(d["z"], d["mu"], d["sigma"], model["mu(z)"])

    if "bao" in datasets:
        d = datasets["bao"]
        chi2 += chi2_bao(d["z"], d["dv_rd"], d["sigma"], model["DVrd(z)"])

    return -0.5 * chi2
