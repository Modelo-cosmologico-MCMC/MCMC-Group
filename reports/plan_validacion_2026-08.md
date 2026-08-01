# Plan de validacion — agosto 2026

Derivado de la revision semanal cerrada el 2026-07-31. Este repositorio
es el entorno de pruebas: todo se valida aqui antes de promoverse al
repositorio principal.

## Estado de las tres prioridades

### 1. Componente de fondo para Cobaya — IMPLEMENTADO

`src/mcmc/cobaya_interface/`: recibe parametros MCMC y produce H(z),
D_M, D_H, D_V y D_L; recupera LCDM plano de forma exacta con
`eps=0, z_trans >= z_max` (verificado contra cuadratura independiente en
`tests/test_cobaya_background.py`). Wrapper `cobaya.theory.Theory`
disponible sin que cobaya sea dependencia obligatoria.
Ver `docs/implementation/cobaya_background.md`.

### 2. Primer ajuste sintetico — IMPLEMENTADO

`scripts/run_synthetic_recovery.py`: BAO+SNe+H(z) sinteticos con verdad
conocida y semilla fija, ajuste MAP + emcee, informe JSON con pulls.

Resultado (seed 20260801): 6/6 parametros dentro de 2 sigma;
round-trip S <-> z con error maximo ~9e-16.

Degeneracion detectada y documentada: la normalizacion H(0)=H0 hace
inobservable la escala conjunta (rho_b0, rho0); el ajuste usa
f_m = rho_b0/(rho_b0+rho0).

### 3. Infraestructura de rigor — IMPLEMENTADO

- Procedencia: `data/provenance.json` + `scripts/register_provenance.py`
  (hash sha256, version, fuente, fecha de todo producto externo;
  verificado en CI). Motivado por la correccion CMB_SPA de SPT-3G.
- Prediccion congelada: `scripts/freeze_predictions.py` sella H(z),
  distancias y w_eff(z) fiduciales antes de DESI DR3 / Rubin.

## Pendiente (proximas iteraciones)

- [ ] Likelihood DESI DR2 con covarianza; Pantheon+ y DES-Y5 por separado
      con marginalizacion de M_B.
- [ ] Matriz completa de combinaciones (sustituir CMB y SNe uno a uno;
      retirar low-z, LRG1, LRG2) — no una unica combinacion favorable.
- [ ] Priors comprimidos CMB (r_d, theta_*) antes de SPT-3G TT/TE/EE.
- [ ] Registrar likelihoods SPT-3G y cadenas corregidas (CMB_SPA fixed)
      en el registro de procedencia al descargarlas.
- [ ] Definicion matematica unica de S (orden / porcentaje / estado /
      variable dinamica) y derivar las demas interpretaciones.
- [ ] Q(z) derivada del formalismo (distinguible de Q = beta H rho_de y
      Q = beta H rho_dm genericas).
- [ ] Protocolo ciego MCV: congelar antes de examinar BIG-SPARC;
      SPARC desarrollo / LITTLE THINGS enanas / THINGS validacion /
      WALLABY test externo, con division por calidad.
- [ ] AIC, BIC y evidencia (PolyChord) cuando el pipeline sea estable.
- [ ] Fase 2 del componente cobaya: espectros CMB via CLASS/CAMB.

## Criterio para nueva version en Zenodo

Solo con al menos uno de: primera likelihood conjunta reproducible;
prediccion galactica MCV explicita; ecuaciones de perturbaciones;
correccion sustancial del formalismo. No por cambios expositivos.
