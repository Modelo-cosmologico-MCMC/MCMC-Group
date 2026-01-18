# Formato de datasets (Bloque II)

## Principio
Cada dataset se define por:
- kind: "hz" | "sne" | "bao"
- CSV con columnas canónicas
- opcional: covarianza completa (cov o cov_inv) en .npy o .npz

El loader canónico es: `mcmc.data.load_dataset(kind, path, cov_path=...)`.

## H(z) (hz)
CSV:
- z
- H
- sigma   (si hay cov completa, sigma puede ignorarse pero se admite)

Covarianza (opcional):
- .npy con matriz NxN (C)
- .npz con clave 'cov' o 'cov_inv'

## Supernovas (sne)
CSV:
- z
- mu
- sigma

Covarianza (opcional):
- sne_cov.npz con 'cov' o 'cov_inv'

Nota: M (offset absoluto) se trata como nuisance en el ajuste.

## BAO (bao)
CSV:
- z
- dv_rd
- sigma

Donde:
- dv_rd = D_V(z)/r_d

Covarianza (opcional):
- bao_cov.npz con 'cov' o 'cov_inv'
