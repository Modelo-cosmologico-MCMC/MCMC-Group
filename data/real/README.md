# Datasets reales (no se versionan en Git)

Esta carpeta está reservada para datasets reales (Pantheon/Pantheon+, BAO BOSS/eBOSS, H(z) chronometers, etc.)
que no se suben al repositorio por tamaño/licencia.

## Formato canónico requerido (CSV)
El pipeline del MCMC usa CSV normalizado por canal:

### H(z) — kind="hz"
CSV con columnas:
- z
- H
- sigma

Opcional: covarianza completa en `*.npy` (NxN) o `*.npz` con clave `cov` o `cov_inv`.

Ejemplo esperado:
- data/real/hz_chronometers.csv
- data/real/hz_chronometers_cov.npz   (opcional)

### Supernovas — kind="sne"
CSV con columnas:
- z
- mu
- sigma

Opcional:
- data/real/sne_pantheon_cov.npz (con `cov` o `cov_inv`)

Nota: el parámetro absoluto M se ajusta como nuisance en el modelo (PR-03).

### BAO — kind="bao"
CSV con columnas:
- z
- dv_rd
- sigma

donde:
- dv_rd = D_V(z)/r_d

Opcional:
- data/real/bao_boss_cov.npz (con `cov` o `cov_inv`)

## Cómo activar datasets reales
1) Copia los archivos a `data/real/`.
2) Edita `src/mcmc/config/defaults.yaml` y cambia `data:`
   - hz:  data/real/hz_chronometers.csv
   - hz_cov: data/real/hz_chronometers_cov.npz  (si existe)
   - sne: data/real/sne_pantheon.csv
   - sne_cov: data/real/sne_pantheon_cov.npz
   - bao: data/real/bao_boss.csv
   - bao_cov: data/real/bao_boss_cov.npz
3) Verifica:
   python scripts/validate_datasets.py

## Covarianzas
Se aceptan:
- `.npy` con matriz NxN llamada directamente
- `.npz` con una de las claves:
  - `cov`     (matriz NxN)
  - `cov_inv` (matriz NxN)

Se recomienda `cov_inv` si N es grande para evitar invertir en runtime.
