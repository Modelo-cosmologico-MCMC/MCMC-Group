from __future__ import annotations

import numpy as np
import emcee


def run_emcee(logprob_fn, x0: np.ndarray, *, nwalkers: int = 32, nsteps: int = 2000, seed: int = 42):
    """
    Ejecuta emcee con inicializacion gaussiana alrededor de x0.

    emcee 3 toma sus movimientos de propuesta del estado GLOBAL legacy
    de numpy (np.random), no de un RNG propio; sembrar solo la bola
    inicial NO reproduce la cadena. np.random.seed(seed) fija el stream
    completo, de modo que la cadena es reproducible entre procesos con
    la misma semilla (misma leccion que cosmology/desi_background_fit.py
    en el repositorio principal; sin ella, los tests de recuperacion
    con cadena corta son intermitentes).
    """
    np.random.seed(seed)                 # stream interno del sampler
    rng = np.random.default_rng(seed)    # bola inicial
    ndim = len(x0)
    p0 = x0 + 1e-3 * rng.standard_normal(size=(nwalkers, ndim))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob_fn)
    sampler.run_mcmc(p0, nsteps, progress=False)
    return sampler
