from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


Kind = Literal["hz", "sne", "bao"]


@dataclass(frozen=True)
class Dataset:
    kind: Kind
    name: str
    z: np.ndarray
    y: np.ndarray
    sigma: np.ndarray | None
    cov: np.ndarray | None
    cov_inv: np.ndarray | None
    meta: dict[str, Any]

    def as_legacy_dict(self) -> dict[str, Any]:
        """
        Compatibilidad con el pipeline actual:
        - hz:  {"z","H","sigma", ...}
        - sne: {"z","mu","sigma", ...}
        - bao: {"z","dv_rd","sigma", ...}
        y añade opcionalmente "cov" y "cov_inv".
        """
        base: dict[str, Any] = {"z": self.z}
        if self.kind == "hz":
            base["H"] = self.y
        elif self.kind == "sne":
            base["mu"] = self.y
        elif self.kind == "bao":
            base["dv_rd"] = self.y
        else:
            raise ValueError("kind desconocido")

        if self.sigma is not None:
            base["sigma"] = self.sigma
        if self.cov is not None:
            base["cov"] = self.cov
        if self.cov_inv is not None:
            base["cov_inv"] = self.cov_inv
        base["name"] = self.name
        base["meta"] = dict(self.meta)
        return base


def _require_columns(df: pd.DataFrame, cols: list[str], *, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset '{name}' missing columns: {missing}. Found: {list(df.columns)}")


def _load_covariance(cov_path: str | None, *, n: int, name: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    if cov_path is None:
        return None, None

    p = Path(cov_path)
    if not p.exists():
        raise FileNotFoundError(f"Covariance file not found: {cov_path}")

    if p.suffix == ".npy":
        arr = np.load(p)
        arr = np.asarray(arr, float)
        if arr.shape != (n, n):
            raise ValueError(f"Covariance shape mismatch for '{name}': {arr.shape} expected {(n,n)}")
        return arr, None

    if p.suffix == ".npz":
        z = np.load(p)
        if "cov" in z:
            cov = np.asarray(z["cov"], float)
            if cov.shape != (n, n):
                raise ValueError(f"cov shape mismatch for '{name}': {cov.shape} expected {(n,n)}")
            return cov, None
        if "cov_inv" in z:
            cov_inv = np.asarray(z["cov_inv"], float)
            if cov_inv.shape != (n, n):
                raise ValueError(f"cov_inv shape mismatch for '{name}': {cov_inv.shape} expected {(n,n)}")
            return None, cov_inv
        raise ValueError(f"NPZ covariance must contain 'cov' or 'cov_inv' for '{name}'")

    raise ValueError(f"Unsupported covariance extension for '{name}': {p.suffix} (use .npy or .npz)")


def _finalize_sigma_from_cov(
    sigma: np.ndarray | None, cov: np.ndarray | None, cov_inv: np.ndarray | None
) -> np.ndarray | None:
    if sigma is not None:
        return sigma
    if cov is not None:
        d = np.diag(cov)
        return np.sqrt(np.maximum(d, 0.0))
    if cov_inv is not None:
        # Si solo tenemos C^{-1}, no podemos recuperar sigmas sin invertir (evitar aquí).
        return None
    return None


def load_dataset(kind: Kind, path: str, *, name: str | None = None, cov_path: str | None = None) -> Dataset:
    """
    Loader robusto con contrato:
      - CSV con columnas requeridas
      - σ (diagonal) opcional
      - cov/cov_inv opcional

    Retorna Dataset (canónico) + método as_legacy_dict() para integrarse con el pipeline actual.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(p)
    ds_name = name if name is not None else p.stem

    if kind == "hz":
        _require_columns(df, ["z", "H", "sigma"], name=ds_name)
        z = df["z"].to_numpy(dtype=float)
        y = df["H"].to_numpy(dtype=float)
        sigma = df["sigma"].to_numpy(dtype=float)
    elif kind == "sne":
        _require_columns(df, ["z", "mu", "sigma"], name=ds_name)
        z = df["z"].to_numpy(dtype=float)
        y = df["mu"].to_numpy(dtype=float)
        sigma = df["sigma"].to_numpy(dtype=float)
    elif kind == "bao":
        _require_columns(df, ["z", "dv_rd", "sigma"], name=ds_name)
        z = df["z"].to_numpy(dtype=float)
        y = df["dv_rd"].to_numpy(dtype=float)
        sigma = df["sigma"].to_numpy(dtype=float)
    else:
        raise ValueError("kind desconocido")

    n = len(z)
    cov, cov_inv = _load_covariance(cov_path, n=n, name=ds_name)
    sigma = _finalize_sigma_from_cov(sigma, cov, cov_inv)

    meta = {
        "path": str(p),
        "cov_path": cov_path,
        "n": n,
        "kind": kind,
    }

    _validate_dataset_arrays(kind, ds_name, z, y, sigma, cov, cov_inv)

    return Dataset(kind=kind, name=ds_name, z=z, y=y, sigma=sigma, cov=cov, cov_inv=cov_inv, meta=meta)


def _validate_dataset_arrays(
    kind: Kind,
    name: str,
    z: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray | None,
    cov: np.ndarray | None,
    cov_inv: np.ndarray | None,
) -> None:
    if z.ndim != 1 or y.ndim != 1:
        raise ValueError(f"Dataset '{name}' arrays must be 1D")
    if len(z) != len(y):
        raise ValueError(f"Dataset '{name}' length mismatch: len(z)={len(z)} len(y)={len(y)}")
    if not np.isfinite(z).all() or not np.isfinite(y).all():
        raise ValueError(f"Dataset '{name}' contains NaN/Inf")
    if np.any(z < 0):
        raise ValueError(f"Dataset '{name}' contains negative z")
    if sigma is not None:
        if sigma.shape != y.shape:
            raise ValueError(f"Dataset '{name}' sigma shape mismatch: {sigma.shape} vs {y.shape}")
        if not np.isfinite(sigma).all():
            raise ValueError(f"Dataset '{name}' sigma contains NaN/Inf")
        if np.any(sigma <= 0):
            raise ValueError(f"Dataset '{name}' sigma must be > 0")

    if cov is not None:
        if cov.shape != (len(y), len(y)):
            raise ValueError(f"Dataset '{name}' cov shape mismatch: {cov.shape}")
        if not np.isfinite(cov).all():
            raise ValueError(f"Dataset '{name}' cov contains NaN/Inf")

    if cov_inv is not None:
        if cov_inv.shape != (len(y), len(y)):
            raise ValueError(f"Dataset '{name}' cov_inv shape mismatch: {cov_inv.shape}")
        if not np.isfinite(cov_inv).all():
            raise ValueError(f"Dataset '{name}' cov_inv contains NaN/Inf")

    # checks mínimos por canal
    if kind == "hz" and np.any(y <= 0):
        raise ValueError(f"Dataset '{name}' H(z) must be > 0")
