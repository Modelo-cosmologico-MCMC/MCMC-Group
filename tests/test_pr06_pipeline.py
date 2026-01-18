from __future__ import annotations

import json
from pathlib import Path

from mcmc.pipeline import load_config, run_pipeline


def test_pipeline_runs(tmp_path: Path) -> None:
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "run_id: test",
                f"out_dir: {tmp_path.as_posix()}",
                "block0: {Mp0: 0.999, Ep0: 0.001, eps: 0.001, S_min: 0.001, S_max: 0.009, dS: 0.001}",
                "block1:",
                "  S_min: 0.010",
                "  S_max: 0.020",
                "  dS: 0.001",
                "  H0: 67.4",
                "  sign: 1.0",
                "block2: {Mp0: 0.99, Mpeq: 0.5, gamma_S: 1.0, kappa: 0.02, alpha_base: 0.6666667, delta_alpha: 0.2, w0: -1.0, delta_w: 0.02}",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    out_json = run_pipeline(cfg)
    assert out_json.exists()


def test_unit_duality_preserved(tmp_path: Path) -> None:
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "run_id: test2",
                f"out_dir: {tmp_path.as_posix()}",
                "block0: {Mp0: 0.999, Ep0: 0.001, eps: 0.001, S_min: 0.001, S_max: 0.009, dS: 0.001}",
                "block1: {S_min: 0.010, S_max: 0.050, dS: 0.001, H0: 67.4, sign: 1.0}",
                "block2: {Mp0: 0.99, Mpeq: 0.5, gamma_S: 1.0, kappa: 0.02, alpha_base: 0.6666667, delta_alpha: 0.2, w0: -1.0, delta_w: 0.02}",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    out_json = run_pipeline(cfg)

    data = json.loads(out_json.read_text(encoding="utf-8"))
    Mp = data["block2"]["Mp_eff"]
    Ep = data["block2"]["Ep_eff"]

    for mp, ep in zip(Mp, Ep):
        assert abs((mp + ep) - 1.0) < 1e-8
        assert 0.0 <= mp <= 1.0
        assert 0.0 <= ep <= 1.0


def test_block0_initial_conditions(tmp_path: Path) -> None:
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "run_id: test3",
                f"out_dir: {tmp_path.as_posix()}",
                "block0: {Mp0: 0.999, Ep0: 0.001, eps: 0.001, S_min: 0.001, S_max: 0.009, dS: 0.001}",
                "block1: {S_min: 0.010, S_max: 0.020, dS: 0.001, H0: 67.4, sign: 1.0}",
                "block2: {Mp0: 0.99, Mpeq: 0.5, gamma_S: 1.0, kappa: 0.02, alpha_base: 0.6666667, delta_alpha: 0.2, w0: -1.0, delta_w: 0.02}",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    run_pipeline(cfg)

    ic_path = tmp_path / "test3" / "block0_initial_conditions.json"
    assert ic_path.exists()

    ic = json.loads(ic_path.read_text(encoding="utf-8"))
    assert "Mp_pre" in ic
    assert "Ep_pre" in ic
    assert abs((ic["Mp_pre"] + ic["Ep_pre"]) - 1.0) < 1e-8


def test_block1_scale_factor_normalized(tmp_path: Path) -> None:
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "run_id: test4",
                f"out_dir: {tmp_path.as_posix()}",
                "block0: {Mp0: 0.999, Ep0: 0.001, eps: 0.001, S_min: 0.001, S_max: 0.009, dS: 0.001}",
                "block1: {S_min: 0.010, S_max: 0.100, dS: 0.001, H0: 67.4, sign: 1.0}",
                "block2: {Mp0: 0.99, Mpeq: 0.5, gamma_S: 1.0, kappa: 0.02, alpha_base: 0.6666667, delta_alpha: 0.2, w0: -1.0, delta_w: 0.02}",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    out_json = run_pipeline(cfg)

    data = json.loads(out_json.read_text(encoding="utf-8"))
    a = data["block1"]["a"]

    # a should end at 1.0 (normalized)
    assert abs(a[-1] - 1.0) < 1e-10

    # a should be increasing (expansion)
    for i in range(1, len(a)):
        assert a[i] >= a[i - 1]
