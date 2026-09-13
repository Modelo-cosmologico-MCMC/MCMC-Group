"""Tests del registro de procedencia de datos externos."""
import json

import pytest

from mcmc.data.provenance import (
    load_registry,
    record_file,
    sha256_of_file,
    verify_registry,
)


@pytest.fixture
def sample_file(tmp_path):
    p = tmp_path / "dataset.csv"
    p.write_text("z,dv_rd,sigma\n0.5,8.9,0.4\n", encoding="utf-8")
    return p


@pytest.fixture
def registry_path(tmp_path):
    return tmp_path / "provenance.json"


class TestRecord:
    def test_record_and_load(self, sample_file, registry_path):
        rec = record_file(
            sample_file,
            source="internal-test",
            version="1.0",
            retrieved="2026-08-01",
            registry_path=registry_path,
        )
        assert rec.sha256 == sha256_of_file(sample_file)
        assert rec.size_bytes == sample_file.stat().st_size

        registry = load_registry(registry_path)
        assert rec.path in registry["records"]
        assert registry["records"][rec.path]["version"] == "1.0"

    def test_missing_file_raises(self, registry_path, tmp_path):
        with pytest.raises(FileNotFoundError):
            record_file(tmp_path / "nope.csv", registry_path=registry_path)


class TestVerify:
    def test_verify_passes_when_unchanged(self, sample_file, registry_path):
        record_file(sample_file, registry_path=registry_path)
        result = verify_registry(registry_path)
        assert result["passed"]
        assert str(sample_file).replace("\\", "/") in result["ok"]

    def test_verify_detects_tampering(self, sample_file, registry_path):
        record_file(sample_file, registry_path=registry_path)
        sample_file.write_text("z,dv_rd,sigma\n0.5,9.9,0.4\n", encoding="utf-8")
        result = verify_registry(registry_path)
        assert not result["passed"]
        assert str(sample_file).replace("\\", "/") in result["changed"]

    def test_verify_detects_missing(self, sample_file, registry_path):
        record_file(sample_file, registry_path=registry_path)
        sample_file.unlink()
        result = verify_registry(registry_path)
        assert not result["passed"]
        assert str(sample_file).replace("\\", "/") in result["missing"]


class TestRepoRegistry:
    def test_committed_registry_verifies(self):
        # El registro versionado del repo debe verificar siempre en CI
        result = verify_registry("data/provenance.json")
        assert result["passed"], (
            f"Registro de procedencia roto: cambiados={result['changed']} "
            f"faltantes={result['missing']}"
        )

    def test_demo_files_registered(self):
        registry = load_registry("data/provenance.json")
        for name in ("hz", "sne", "bao"):
            assert f"data/demo/{name}.csv" in registry["records"]


class TestInvalidRegistry:
    def test_invalid_registry_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"foo": 1}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_registry(bad)
