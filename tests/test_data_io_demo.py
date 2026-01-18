import numpy as np
from mcmc.data.io import load_dataset


def test_load_demo_hz():
    ds = load_dataset("hz", "data/demo/hz.csv", name="demo_hz")
    assert ds.kind == "hz"
    assert len(ds.z) == len(ds.y)
    assert np.isfinite(ds.z).all()
    assert np.isfinite(ds.y).all()
    assert ds.sigma is not None
    assert (ds.sigma > 0).all()


def test_load_demo_sne():
    ds = load_dataset("sne", "data/demo/sne.csv", name="demo_sne")
    assert ds.kind == "sne"
    assert len(ds.z) == len(ds.y)
    assert ds.sigma is not None
    assert (ds.sigma > 0).all()


def test_load_demo_bao():
    ds = load_dataset("bao", "data/demo/bao.csv", name="demo_bao")
    assert ds.kind == "bao"
    assert len(ds.z) == len(ds.y)
    assert ds.sigma is not None
    assert (ds.sigma > 0).all()
