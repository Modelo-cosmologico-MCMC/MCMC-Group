# Mapeo ecuacion -> modulo

Este documento establece la correspondencia entre las ecuaciones del tratado
teorico y los modulos de implementacion.

## Bloque 0 (Pre-geometrico)

| Concepto | Ecuacion | Modulo |
|----------|----------|--------|
| Condiciones iniciales | Mp_pre, Ep_pre, phi_pre | `src/mcmc/pregeom/s0_state.py` |
| Contrato de salida | initial_conditions.json | `scripts/run_pregeom_export.py` |

## Bloque I (Nucleo ontologico)

| Concepto | Ecuacion | Modulo |
|----------|----------|--------|
| Rejilla S | S in [0.010, 1.001], dS=1e-3 | `src/mcmc/core/s_grid.py` |
| Sellos | S1, S2, S3, S4 | `src/mcmc/core/s_grid.py` |
| C(S) expansion | d(ln a)/dS = C(S) | `src/mcmc/core/cronoshapes.py` |
| T(S) cronificacion | cadencia base | `src/mcmc/core/cronoshapes.py` |
| Phi_ten(S) tensional | campo de Adrian | `src/mcmc/core/cronoshapes.py` |
| N(S) lapse | N = exp(Phi_ten) | `src/mcmc/core/cronoshapes.py` |
| Integracion a(S) | a(S4)=1, hacia atras | `src/mcmc/core/background.py` |
| Integracion t(S) | dt/dS = T*N | `src/mcmc/core/background.py` |
| H(S) | H = H0 * C/C(S4) | `src/mcmc/core/background.py` |
| z(S) | z = 1/a - 1 | `src/mcmc/core/background.py` |
| Checks coherencia | monotonicidad, positividad | `src/mcmc/core/checks.py` |

## Bloque II (Cosmologia efectiva)

| Concepto | Ecuacion | Modulo |
|----------|----------|--------|
| rho_bar(z) | rho_b0 * (1+z)^3 | `src/mcmc/core/friedmann_effective.py` |
| rho_id(z) transicion | por tramos en z_trans | `src/mcmc/channels/rho_id_refined.py` |
| H(z) efectivo | H0 * sqrt(rho_bar + rho_id) | `src/mcmc/core/friedmann_effective.py` |
| r(z) comoving | integral c/H | `src/mcmc/observables/distances.py` |
| D_A(z) angular | r/(1+z) | `src/mcmc/observables/bao_distances.py` |
| D_V(z) volumen | [(1+z)^2 D_A^2 cz/H]^(1/3) | `src/mcmc/observables/bao_distances.py` |
| d_L(z) luminosidad | (1+z) * r | `src/mcmc/observables/distances.py` |
| mu(z) modulo | 5*log10(d_L) + 25 | `src/mcmc/observables/distances.py` |
| chi2_BAO | suma ponderada | `src/mcmc/observables/bao.py` |
| chi2_Hz | suma ponderada | `src/mcmc/observables/hz.py` |
| chi2_SNe | suma ponderada | `src/mcmc/observables/sne.py` |
| chi2_total | BAO + Hz + SNe | `src/mcmc/observables/likelihoods.py` |
| AIC, BIC | criterios informacion | `src/mcmc/observables/info_criteria.py` |

## Inferencia

| Concepto | Modulo |
|----------|--------|
| Priors uniformes/gaussianos | `src/mcmc/inference/priors.py` |
| Sampler emcee | `src/mcmc/inference/emcee_fit.py` |
| Postproceso cadenas | `src/mcmc/inference/postprocess.py` |
| Outputs/diagnosticos | `src/mcmc/inference/outputs.py` |

## N-body (Cronos toy)

| Concepto | Modulo |
|----------|--------|
| Integrador KDK | `src/mcmc/nbody/kdk.py` |
| Timestep Cronos | `src/mcmc/nbody/crono_step.py` |
| Poisson/aceleracion | `src/mcmc/nbody/poisson.py` |
