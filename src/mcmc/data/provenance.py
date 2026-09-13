"""Registro de procedencia de productos de datos externos.

Motivacion (informe semanal, punto SPT-3G / CMB_SPA): un pipeline que use
cadenas o likelihoods equivocadas puede producir tensiones o concordancias
falsas. Todo producto externo (mapas, likelihoods, cadenas, catalogos)
debe registrarse con hash sha256, version, fuente y fecha de descarga, y
verificarse antes de cada uso.

Registro: JSON versionado en data/provenance.json.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

REGISTRY_VERSION = 1
DEFAULT_REGISTRY = Path("data/provenance.json")


def sha256_of_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """sha256 de un fichero leyendo por bloques."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ProvenanceRecord:
    """Ficha de procedencia de un producto externo.

    Attributes:
        path: Ruta relativa al repositorio
        sha256: Hash del contenido en el momento del registro
        size_bytes: Tamano del fichero
        source: Origen (URL, DOI, 'internal-demo', ...)
        version: Version del producto segun el proveedor
        retrieved: Fecha de obtencion (ISO 8601)
        notes: Correcciones conocidas, erratas, etc.
    """
    path: str
    sha256: str
    size_bytes: int
    source: str = ""
    version: str = ""
    retrieved: str = ""
    notes: str = ""


def load_registry(registry_path: str | Path = DEFAULT_REGISTRY) -> dict:
    p = Path(registry_path)
    if not p.exists():
        return {"registry_version": REGISTRY_VERSION, "records": {}}
    data = json.loads(p.read_text(encoding="utf-8"))
    if "records" not in data:
        raise ValueError(f"Registro invalido (sin 'records'): {registry_path}")
    return data


def save_registry(registry: dict, registry_path: str | Path = DEFAULT_REGISTRY) -> None:
    p = Path(registry_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record_file(
    path: str | Path,
    *,
    source: str = "",
    version: str = "",
    retrieved: str = "",
    notes: str = "",
    registry_path: str | Path = DEFAULT_REGISTRY,
) -> ProvenanceRecord:
    """Registra (o actualiza) la procedencia de un fichero.

    Returns:
        La ficha registrada.

    Raises:
        FileNotFoundError: si el fichero no existe.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe: {path}")

    rec = ProvenanceRecord(
        path=str(p).replace("\\", "/"),
        sha256=sha256_of_file(p),
        size_bytes=p.stat().st_size,
        source=source,
        version=version,
        retrieved=retrieved,
        notes=notes,
    )
    registry = load_registry(registry_path)
    registry["records"][rec.path] = asdict(rec)
    save_registry(registry, registry_path)
    return rec


def verify_registry(registry_path: str | Path = DEFAULT_REGISTRY) -> dict:
    """Verifica todos los ficheros registrados contra su hash.

    Returns:
        {'ok': [...], 'changed': [...], 'missing': [...], 'passed': bool}
    """
    registry = load_registry(registry_path)
    ok, changed, missing = [], [], []
    for rel_path, rec in sorted(registry["records"].items()):
        p = Path(rel_path)
        if not p.exists():
            missing.append(rel_path)
            continue
        if sha256_of_file(p) == rec["sha256"]:
            ok.append(rel_path)
        else:
            changed.append(rel_path)
    return {
        "ok": ok,
        "changed": changed,
        "missing": missing,
        "passed": not changed and not missing,
    }
