"""Smoke tests for PR-07 unified execution pipeline."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mcmc.pipeline import run_from_config, load_yaml


@pytest.fixture
def demo_data_paths() -> dict[str, str]:
    """Return paths to demo datasets."""
    return {
        "hz": "data/demo/hz.csv",
        "sne": "data/demo/sne.csv",
        "bao": "data/demo/bao.csv",
    }


def _write_config(path: Path, cfg: dict) -> Path:
    """Write config dict to YAML file."""
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


class TestPipelineEvaluate:
    """Tests for evaluate mode."""

    def test_pipeline_evaluate_effective(self, tmp_path: Path, demo_data_paths: dict[str, str]) -> None:
        """Test evaluate mode with effective backend."""
        cfg = {
            "run": {
                "run_id": "test_eval_eff",
                "outdir": str(tmp_path),
                "mode": "evaluate",
            },
            "model": {"backend": "effective"},
            "data": demo_data_paths,
            "effective": {
                "H0": 67.4,
                "rho_b0": 0.30,
                "rd": 147.0,
                "M": -19.3,
                "rho_id": {
                    "rho0": 0.70,
                    "z_trans": 1.0,
                    "eps": 0.05,
                },
            },
        }
        config_path = _write_config(tmp_path / "config.yaml", cfg)

        out = run_from_config(config_path)

        assert out.outdir.exists()
        assert (out.outdir / "loglike.txt").exists()
        assert (out.outdir / "summary.txt").exists()
        assert (out.outdir / "config_used.yaml").exists()
        assert out.loglike < 0  # Should be negative (log-likelihood)
        assert out.chain_path is None  # No chain in evaluate mode
        assert out.logp_path is None

    def test_pipeline_evaluate_block1(self, tmp_path: Path, demo_data_paths: dict[str, str]) -> None:
        """Test evaluate mode with block1 backend."""
        cfg = {
            "run": {
                "run_id": "test_eval_b1",
                "outdir": str(tmp_path),
                "mode": "evaluate",
            },
            "model": {"backend": "block1"},
            "data": demo_data_paths,
            "block1": {
                "H0": 67.4,
                "rd": 147.0,
                "M": -19.3,
            },
        }
        config_path = _write_config(tmp_path / "config.yaml", cfg)

        out = run_from_config(config_path)

        assert out.outdir.exists()
        assert (out.outdir / "loglike.txt").exists()
        assert out.loglike < 0

    @pytest.mark.skip(reason="unified backend not yet implemented in builder")
    def test_pipeline_evaluate_unified(self, tmp_path: Path, demo_data_paths: dict[str, str]) -> None:
        """Test evaluate mode with unified backend."""
        cfg = {
            "run": {
                "run_id": "test_eval_uni",
                "outdir": str(tmp_path),
                "mode": "evaluate",
            },
            "model": {"backend": "unified"},
            "data": demo_data_paths,
            "unified": {
                "H0": 67.4,
                "rd": 147.0,
                "M": -19.3,
                "rho_b0": 0.30,
                "lat_enabled": False,
            },
        }
        config_path = _write_config(tmp_path / "config.yaml", cfg)

        out = run_from_config(config_path)

        assert out.outdir.exists()
        assert (out.outdir / "loglike.txt").exists()
        assert out.loglike < 0


class TestPipelineFit:
    """Tests for fit mode (emcee)."""

    def test_fit_mode_effective_short(self, tmp_path: Path, demo_data_paths: dict[str, str]) -> None:
        """Test fit mode with effective backend (short run for CI)."""
        cfg = {
            "run": {
                "run_id": "test_fit_short",
                "outdir": str(tmp_path),
                "mode": "fit",
                "seed": 42,
                "nwalkers": 16,  # Must be >= 2*ndim (7 params -> 14 minimum)
                "nsteps": 10,  # Very short for smoke test
                "burn_frac": 0.10,
                "thin": 1,
            },
            "model": {"backend": "effective"},
            "data": demo_data_paths,
            "effective": {
                "H0": 67.4,
                "rho_b0": 0.30,
                "rd": 147.0,
                "M": -19.3,
                "rho_id": {
                    "rho0": 0.70,
                    "z_trans": 1.0,
                    "eps": 0.05,
                },
            },
        }
        config_path = _write_config(tmp_path / "config.yaml", cfg)

        out = run_from_config(config_path)

        assert out.outdir.exists()
        assert out.chain_path is not None
        assert out.chain_path.exists()
        assert out.logp_path is not None
        assert out.logp_path.exists()
        assert (out.outdir / "summary.txt").exists()

    def test_fit_mode_block1_not_supported(self, tmp_path: Path, demo_data_paths: dict[str, str]) -> None:
        """Test that fit mode with non-effective backend returns gracefully."""
        cfg = {
            "run": {
                "run_id": "test_fit_b1",
                "outdir": str(tmp_path),
                "mode": "fit",
            },
            "model": {"backend": "block1"},
            "data": demo_data_paths,
            "block1": {
                "H0": 67.4,
                "rd": 147.0,
                "M": -19.3,
            },
        }
        config_path = _write_config(tmp_path / "config.yaml", cfg)

        out = run_from_config(config_path)

        # Should complete but without chain
        assert out.outdir.exists()
        assert out.chain_path is None
        summary_text = out.summary_path.read_text()
        assert "ERROR" in summary_text or "evaluate" in summary_text.lower()


class TestLoadYaml:
    """Tests for load_yaml utility."""

    def test_load_yaml_basic(self, tmp_path: Path) -> None:
        """Test basic YAML loading."""
        cfg = {"key": "value", "nested": {"a": 1, "b": 2}}
        path = tmp_path / "test.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        loaded = load_yaml(path)
        assert loaded == cfg

    def test_load_yaml_str_path(self, tmp_path: Path) -> None:
        """Test YAML loading with string path."""
        cfg = {"run": {"mode": "evaluate"}}
        path = tmp_path / "test2.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        loaded = load_yaml(str(path))
        assert loaded["run"]["mode"] == "evaluate"


class TestConfigUsed:
    """Tests for config persistence."""

    def test_config_persisted(self, tmp_path: Path, demo_data_paths: dict[str, str]) -> None:
        """Test that used config is saved to output directory."""
        cfg = {
            "run": {
                "run_id": "test_persist",
                "outdir": str(tmp_path),
                "mode": "evaluate",
            },
            "model": {"backend": "effective"},
            "data": demo_data_paths,
            "effective": {
                "H0": 70.0,
                "rho_b0": 0.28,
                "rd": 145.0,
                "M": -19.2,
                "rho_id": {"rho0": 0.72, "z_trans": 0.8, "eps": 0.03},
            },
        }
        config_path = _write_config(tmp_path / "config.yaml", cfg)

        out = run_from_config(config_path)

        config_used_path = out.outdir / "config_used.yaml"
        assert config_used_path.exists()
        loaded = load_yaml(config_used_path)
        assert loaded["effective"]["H0"] == 70.0
        assert loaded["effective"]["rho_b0"] == 0.28
