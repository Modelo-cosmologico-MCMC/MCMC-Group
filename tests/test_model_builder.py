import numpy as np
from mcmc.models.builder import build_model_from_config


def test_build_effective_model_contract():
    cfg = {
        "model": {"backend": "effective"},
        "effective": {
            "H0": 67.4,
            "rho_b0": 0.3,
            "rd": 147.0,
            "M": -19.3,
            "rho_id": {"rho0": 0.7, "z_trans": 1.0, "eps": 0.05},
        },
    }
    m = build_model_from_config(cfg)
    z = np.array([0.1, 0.5])
    assert np.isfinite(m["H(z)"](z)).all()
    assert np.isfinite(m["mu(z)"](z)).all()
    assert np.isfinite(m["DVrd(z)"](z)).all()


def test_build_block1_model_contract():
    cfg = {
        "model": {"backend": "block1"},
        "block1": {"H0": 67.4, "rd": 147.0, "M": -19.3},
    }
    m = build_model_from_config(cfg)
    z = np.array([0.1, 0.5])
    assert np.isfinite(m["H(z)"](z)).all()
    assert np.isfinite(m["mu(z)"](z)).all()
    assert np.isfinite(m["DVrd(z)"](z)).all()
