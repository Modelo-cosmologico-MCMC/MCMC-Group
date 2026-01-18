# Backends de modelo (PR-05)

## Objetivo
Unificar el contrato observacional:
- H(z)
- mu(z)
- DV/rd(z)

de forma que el likelihood sea agnóstico al backend.

## Backend "effective" (Bloque II)
- H(z) proviene del cierre ρ_bar(z)+ρ_id(z) normalizado a H(0)=H0.
- Ajuste con emcee en scripts/run_fit_emcee.py (por defecto).

## Backend "block1" (Bloque I)
- Se resuelve solve_background(S) y se obtiene z(S), H(S).
- Se construye H(z) por interpolación monotónica (clamp en rango).
- Se evalúa el likelihood (por ahora sin emcee de shapes):
  python scripts/run_fit_emcee.py --evaluate-only

## Configuración
En src/mcmc/config/defaults.yaml:
- model.backend: effective | block1
