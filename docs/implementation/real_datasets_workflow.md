# Workflow de integración de datasets reales (PR-04b)

## Objetivo
Mantener:
- CI en verde con `data/demo/*`
- Reproducibilidad local con `data/real/*` sin versionar

## Conversión al formato canónico
El pipeline NO acepta formatos originales directamente. Debes convertir a CSV simple con columnas requeridas.

### SNe (Pantheon/Pantheon+)
Convertir a:
- z, mu, sigma

Si tienes magnitudes (mB) y correcciones, debes construir mu y sigma siguiendo el esquema del dataset.
En este repo, el modelo ajusta un parámetro nuisance `M`, por lo que mu puede ser relativo mientras sea consistente.

### BAO (BOSS/eBOSS)
Convertir a:
- z, dv_rd, sigma

Si el dataset reporta otras combinaciones (DM/rd, DH/rd, etc.), se recomienda:
- o bien convertir a DV/rd,
- o extender el repo con un segundo tipo BAO (futuro PR) para soportar DM/DH por separado.

### H(z) (chronometers)
Convertir a:
- z, H, sigma

## Covarianzas
- Si dispones de matriz completa: usa .npz con `cov_inv` si puedes.
- Si no, usa diagonal con sigma.

## Validación
- python scripts/validate_datasets.py
- python scripts/run_fit_emcee.py --nsteps 200 --nwalkers 32 --outdir results/fit_real
