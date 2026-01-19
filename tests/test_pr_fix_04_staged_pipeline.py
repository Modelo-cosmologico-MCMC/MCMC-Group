"""Tests for PR-FIX-04: Staged pipeline with coherent outputs.

Verifies:
1. Pipeline produces separate pre-BB and post-BB outputs
2. Output files have correct regime labeling
3. Boundary conditions are properly extracted
4. Both stages can run independently
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import yaml

from mcmc.pipeline.staged import (
    run_staged_pipeline,
    _run_stage_prebb,
    _run_stage_postbb,
)
from mcmc.core.ontology import THRESHOLDS


def _make_test_config(outdir: Path, mode: str = "full") -> Path:
    """Create a minimal test config file."""
    cfg = {
        "run": {
            "run_id": "test_staged",
            "outdir": str(outdir),
            "mode": mode,
        },
        "block0": {
            "Mp0": 0.999,
            "Ep0": 0.001,
            "S_min": 0.001,
            "S_max": 0.009,
        },
        "block1": {
            "S_min": 0.010,
            "S_max": 1.001,
            "H0": 67.4,
        },
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
    config_path = outdir / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return config_path


class TestStagedPipelineFull:
    """Test full staged pipeline."""

    def test_full_mode_creates_both_stages(self) -> None:
        """Full mode produces both pre-BB and post-BB outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            config_path = _make_test_config(outdir, mode="full")

            result = run_staged_pipeline(config_path)

            assert result.prebb_path is not None
            assert result.postbb_path is not None
            assert result.boundary_path is not None
            assert result.summary_path.exists()

    def test_full_mode_output_structure(self) -> None:
        """Full mode output files have correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            config_path = _make_test_config(outdir, mode="full")

            result = run_staged_pipeline(config_path)

            # Check pre-BB output
            prebb = json.loads(result.prebb_path.read_text())
            assert prebb["regime"] == "pre-BB"
            assert "S_range" in prebb
            assert "t_BB" in prebb
            assert "block0" in prebb
            assert "block1" in prebb

            # Check post-BB output
            postbb = json.loads(result.postbb_path.read_text())
            assert postbb["regime"] == "post-BB"
            assert "H0" in postbb
            assert "t0" in postbb
            assert "observables" in postbb


class TestStagedPipelineOntological:
    """Test ontological (pre-BB only) mode."""

    def test_ontological_mode_only_prebb(self) -> None:
        """Ontological mode only produces pre-BB output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            config_path = _make_test_config(outdir, mode="ontological")

            result = run_staged_pipeline(config_path)

            assert result.prebb_path is not None
            assert result.postbb_path is None
            assert result.boundary_path is not None

    def test_boundary_conditions_extracted(self) -> None:
        """Boundary conditions file contains correct data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            config_path = _make_test_config(outdir, mode="ontological")

            result = run_staged_pipeline(config_path)

            bc = json.loads(result.boundary_path.read_text())
            assert "a_rel_BB" in bc
            assert "t_BB" in bc
            assert "Mp_pre" in bc
            assert "Ep_pre" in bc

            # t_BB should be 0 (Chronos anchor)
            assert abs(bc["t_BB"]) < 1e-3


class TestStagedPipelineEvaluate:
    """Test evaluate (post-BB only) mode."""

    def test_evaluate_mode_only_postbb(self) -> None:
        """Evaluate mode only produces post-BB output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            config_path = _make_test_config(outdir, mode="evaluate")

            result = run_staged_pipeline(config_path)

            assert result.prebb_path is None
            assert result.postbb_path is not None

    def test_postbb_observables_valid(self) -> None:
        """Post-BB observables are physically valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            config_path = _make_test_config(outdir, mode="evaluate")

            result = run_staged_pipeline(config_path)

            postbb = json.loads(result.postbb_path.read_text())
            obs = postbb["observables"]

            # H(z=0) should equal H0
            assert abs(obs["H"][0] - postbb["H0"]) < 1e-6

            # H should increase with z (for z < few)
            assert obs["H"][-1] > obs["H"][0]


class TestStagedPipelineOutput:
    """Test output file consistency."""

    def test_summary_contains_regime_info(self) -> None:
        """Summary file contains regime information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            config_path = _make_test_config(outdir, mode="full")

            result = run_staged_pipeline(config_path)

            summary = result.summary_path.read_text()
            assert "Stage 1: Pre-BB" in summary
            assert "Stage 2: Post-BB" in summary
            assert str(THRESHOLDS.S_BB) in summary

    def test_config_saved(self) -> None:
        """Config used is saved in output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            config_path = _make_test_config(outdir, mode="full")

            result = run_staged_pipeline(config_path)

            config_saved = result.outdir / "config_used.yaml"
            assert config_saved.exists()


class TestStageHelpers:
    """Test stage helper functions directly."""

    def test_run_stage_prebb(self) -> None:
        """_run_stage_prebb produces valid result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            cfg = {
                "block0": {},
                "block1": {"S_max": THRESHOLDS.S_BB},
            }

            result = _run_stage_prebb(cfg, outdir)

            assert result.S.max() <= THRESHOLDS.S_BB + 1e-6
            assert abs(result.t_BB) < 1e-3

    def test_run_stage_postbb(self) -> None:
        """_run_stage_postbb produces valid result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            cfg = {
                "effective": {
                    "H0": 67.4,
                    "rho_b0": 0.30,
                    "rd": 147.0,
                    "M": -19.3,
                    "rho_id": {"rho0": 0.70, "z_trans": 1.0, "eps": 0.05},
                },
            }

            result = _run_stage_postbb(cfg, outdir)

            assert result.H0 == 67.4
            assert result.t0 > 0
