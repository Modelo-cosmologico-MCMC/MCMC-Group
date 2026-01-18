from __future__ import annotations

from .hz import chi2_hz
from .sne import chi2_sne
from .bao import chi2_bao


def loglike_total(datasets: dict, model: dict) -> float:
    chi2 = 0.0

    if "hz" in datasets:
        d = datasets["hz"]
        chi2 += chi2_hz(
            d["z"], d["H"], d.get("sigma"), model["H(z)"],
            cov=d.get("cov"), cov_inv=d.get("cov_inv"),
        )

    if "sne" in datasets:
        d = datasets["sne"]
        chi2 += chi2_sne(
            d["z"], d["mu"], d.get("sigma"), model["mu(z)"],
            cov=d.get("cov"), cov_inv=d.get("cov_inv"),
        )

    if "bao" in datasets:
        d = datasets["bao"]
        chi2 += chi2_bao(
            d["z"], d["dv_rd"], d.get("sigma"), model["DVrd(z)"],
            cov=d.get("cov"), cov_inv=d.get("cov_inv"),
        )

    return -0.5 * chi2
