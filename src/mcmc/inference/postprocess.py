from __future__ import annotations

import numpy as np


def summarize_chain(chain: np.ndarray, burn: int = 0, thin: int = 1) -> dict:
    """
    chain: (nsteps, nwalkers, ndim) o (nwalkers, nsteps, ndim)
    """
    arr = np.asarray(chain)
    if arr.ndim != 3:
        raise ValueError("chain debe ser 3D.")
    # normaliza forma a (nsteps, nwalkers, ndim)
    if arr.shape[0] < arr.shape[1]:
        arr = np.transpose(arr, (1, 0, 2))

    arr = arr[burn::thin]
    flat = arr.reshape(-1, arr.shape[-1])
    q16, q50, q84 = np.percentile(flat, [16, 50, 84], axis=0)
    return {"p16": q16, "p50": q50, "p84": q84}
