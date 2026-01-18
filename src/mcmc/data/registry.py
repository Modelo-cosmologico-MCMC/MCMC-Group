from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mcmc.data.io import Kind


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: Kind
    path: str
    cov_path: str | None = None
    description: str = ""
    version: str = "0"
    meta: dict[str, Any] = field(default_factory=dict)


# Registry mínimo (se amplía en PR-04/PR-04b con reales cuando los subas o referencies)
_REGISTRY: dict[str, DatasetSpec] = {
    # DEMO (CI)
    "demo_hz": DatasetSpec(
        name="demo_hz",
        kind="hz",
        path="data/demo/hz.csv",
        cov_path=None,
        description="Demo H(z) with diagonal sigma (CI-safe).",
        version="0.1",
        meta={"source": "internal-demo"},
    ),
    "demo_sne": DatasetSpec(
        name="demo_sne",
        kind="sne",
        path="data/demo/sne.csv",
        cov_path=None,
        description="Demo SNe mu(z) with diagonal sigma (CI-safe).",
        version="0.1",
        meta={"source": "internal-demo"},
    ),
    "demo_bao": DatasetSpec(
        name="demo_bao",
        kind="bao",
        path="data/demo/bao.csv",
        cov_path=None,
        description="Demo BAO DV/rd with diagonal sigma (CI-safe).",
        version="0.1",
        meta={"source": "internal-demo"},
    ),

    # PLACEHOLDERS (reales) — no incluidos en repo por tamaño/licencia.
    # Se activan cuando el usuario añada los ficheros en data/real/* o indique rutas.
    "real_hz_chronometers": DatasetSpec(
        name="real_hz_chronometers",
        kind="hz",
        path="data/real/hz_chronometers.csv",
        cov_path=None,
        description="Cosmic chronometers H(z) compilation (user-provided).",
        version="TBD",
        meta={"note": "Add file to data/real/ and update version/meta."},
    ),
    "real_sne_pantheon": DatasetSpec(
        name="real_sne_pantheon",
        kind="sne",
        path="data/real/sne_pantheon.csv",
        cov_path="data/real/sne_pantheon_cov.npz",
        description="Pantheon/Pantheon+ SNe with covariance (user-provided).",
        version="TBD",
        meta={"note": "Provide cov.npz with 'cov' or 'cov_inv' key."},
    ),
    "real_bao_boss": DatasetSpec(
        name="real_bao_boss",
        kind="bao",
        path="data/real/bao_boss.csv",
        cov_path="data/real/bao_boss_cov.npz",
        description="BAO BOSS/eBOSS DV/rd with covariance (user-provided).",
        version="TBD",
        meta={"note": "Provide DV/rd values and covariance."},
    ),
}


def list_datasets() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_spec(name: str) -> DatasetSpec:
    if name not in _REGISTRY:
        raise KeyError(f"Dataset '{name}' not found. Available: {list_datasets()}")
    return _REGISTRY[name]
