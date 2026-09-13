"""Generacion de datos sinteticos BAO / SNe / H(z) con verdad conocida.

Los datos se generan con BackgroundCalculator (el mismo motor que usara
la inferencia), ruido gaussiano con semilla fija y verdad registrada en
JSON con hash sha256 de cada fichero generado. Esto permite:

  - comprobar que el pipeline recupera los parametros inyectados;
  - detectar degeneraciones (p.ej. la reescala conjunta rho_b0/rho0);
  - dejar rastro reproducible (semilla + hash) de cada realizacion.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from mcmc.cobaya_interface.background import BackgroundCalculator, BackgroundParams


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


@dataclass(frozen=True)
class SyntheticSpec:
    """Especificacion de una realizacion sintetica.

    Attributes:
        truth: Parametros verdaderos del fondo
        seed: Semilla del generador de ruido
        z_hz / z_sne / z_bao: Redshifts de muestreo por observable
        sigma_hz_frac: Error relativo en H(z)
        sigma_sne: Error absoluto en mu [mag]
        sigma_bao_frac: Error relativo en DV/rd
    """
    truth: BackgroundParams = field(default_factory=BackgroundParams)
    seed: int = 20260801
    z_hz: tuple = tuple(np.round(np.linspace(0.07, 2.0, 30), 4))
    z_sne: tuple = tuple(np.round(np.geomspace(0.01, 2.3, 60), 4))
    z_bao: tuple = (0.106, 0.15, 0.32, 0.38, 0.51, 0.57, 0.61, 0.70, 0.85, 1.48, 2.33)
    sigma_hz_frac: float = 0.03
    sigma_sne: float = 0.12
    sigma_bao_frac: float = 0.015


def generate_synthetic_datasets(
    spec: SyntheticSpec,
    outdir: str | Path = "data/synthetic",
    *,
    write_truth: bool = True,
) -> dict:
    """Genera hz.csv, sne.csv y bao.csv sinteticos y truth.json.

    Args:
        spec: Especificacion (verdad, semilla, muestreo, errores)
        outdir: Directorio de salida
        write_truth: Si True escribe truth.json con verdad + hashes

    Returns:
        Dict con claves 'hz', 'sne', 'bao' en el formato legacy del
        pipeline (z, H/mu/dv_rd, sigma) y 'truth' con los parametros.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(spec.seed)
    calc = BackgroundCalculator(spec.truth)

    # H(z)
    z_hz = np.asarray(spec.z_hz, float)
    H_true = calc.H(z_hz)
    sig_hz = spec.sigma_hz_frac * H_true
    H_obs = H_true + rng.standard_normal(z_hz.size) * sig_hz

    # SNe mu(z)
    z_sne = np.asarray(spec.z_sne, float)
    mu_true = calc.mu(z_sne)
    sig_sne = np.full(z_sne.size, spec.sigma_sne)
    mu_obs = mu_true + rng.standard_normal(z_sne.size) * sig_sne

    # BAO DV/rd
    z_bao = np.asarray(spec.z_bao, float)
    dvrd_true = calc.DV_rd(z_bao)
    sig_bao = spec.sigma_bao_frac * dvrd_true
    dvrd_obs = dvrd_true + rng.standard_normal(z_bao.size) * sig_bao

    files = {
        "hz": (outdir / "hz.csv",
               pd.DataFrame({"z": z_hz, "H": H_obs, "sigma": sig_hz})),
        "sne": (outdir / "sne.csv",
                pd.DataFrame({"z": z_sne, "mu": mu_obs, "sigma": sig_sne})),
        "bao": (outdir / "bao.csv",
                pd.DataFrame({"z": z_bao, "dv_rd": dvrd_obs, "sigma": sig_bao})),
    }
    hashes = {}
    for key, (path, df) in files.items():
        df.to_csv(path, index=False, float_format="%.8f")
        hashes[key] = {"path": str(path), "sha256": _sha256(path)}

    if write_truth:
        truth_payload = {
            "generator": "mcmc.validation.synthetic",
            "seed": spec.seed,
            "truth": asdict(spec.truth),
            "noise": {
                "sigma_hz_frac": spec.sigma_hz_frac,
                "sigma_sne": spec.sigma_sne,
                "sigma_bao_frac": spec.sigma_bao_frac,
            },
            "files": hashes,
        }
        (outdir / "truth.json").write_text(
            json.dumps(truth_payload, indent=2), encoding="utf-8"
        )

    return {
        "hz": {"z": z_hz, "H": H_obs, "sigma": sig_hz},
        "sne": {"z": z_sne, "mu": mu_obs, "sigma": sig_sne},
        "bao": {"z": z_bao, "dv_rd": dvrd_obs, "sigma": sig_bao},
        "truth": spec.truth,
        "hashes": hashes,
    }
