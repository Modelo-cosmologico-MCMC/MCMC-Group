# Componente de fondo estilo Cobaya

Implementa la Prioridad 1 del plan de validacion: un componente teorico
que recibe parametros MCMC y devuelve `H(z)`, `D_M`, `D_H`, `D_V` y `D_L`,
recuperando LCDM en un limite verificable.

## Arquitectura

```
mcmc/cobaya_interface/
├── background.py       # BackgroundCalculator (motor numerico, sin cobaya)
├── lcdm_reference.py   # LCDM independiente (quad) para verificacion cruzada
└── theory.py           # Wrapper cobaya.theory.Theory (cobaya opcional)
```

El motor numerico no depende de cobaya: todo el contenido fisico es
testeable con la suite estandar. El wrapper `theory.get_theory_class()`
solo importa cobaya cuando se invoca.

## Uso basico

```python
from mcmc.cobaya_interface import BackgroundCalculator, BackgroundParams

p = BackgroundParams(H0=67.4, rho_b0=0.30, rho0=0.70,
                     z_trans=1.0, eps=0.05, rd=147.0, M=-19.3)
calc = BackgroundCalculator(p)

z = [0.32, 0.51, 1.48]
calc.H(z)      # H(z) [km/s/Mpc]
calc.D_M(z)    # distancia comovil transversal [Mpc]
calc.D_H(z)    # c / H(z) [Mpc]
calc.D_V(z)    # distancia de volumen BAO [Mpc]
calc.D_L(z)    # distancia de luminosidad [Mpc]
calc.mu(z)     # modulo de distancia SNe
calc.DV_rd(z)  # observable BAO adimensional
calc.w_eff_id(z)  # ecuacion de estado efectiva del canal indeterminado
calc.table(z)  # tabla completa
```

## Limite LCDM verificable

Con `eps = 0` y `z_trans >= z_max`, `rho_id(z)` es exactamente constante y

```
E^2(z) = [rho_b0 (1+z)^3 + rho0] / [rho_b0 + rho0]
```

coincide con LCDM plano de `Omega_m = rho_b0 / (rho_b0 + rho0)`:

```python
calc = BackgroundCalculator.lcdm(H0=67.4, Omega_m=0.315)
```

`tests/test_cobaya_background.py` verifica este limite contra una
implementacion independiente por cuadratura adaptativa
(`lcdm_reference.LCDMReference`), con tolerancias de 1e-6 relativas en
distancias y `w_eff_id = -1` exacto.

## Invariantes de estabilidad

`BackgroundCalculator.validate()` se ejecuta en la construccion:

- `H(z)` finito y `H^2(z) > 0` en toda la malla;
- normalizacion exacta `H(0) = H0`;
- `D_C(z)` estrictamente creciente.

Un punto de parametros que viole estos invariantes lanza `ValueError`
(en el wrapper de cobaya el punto se rechaza devolviendo `False`).

## Degeneracion conocida (documentada)

Por la normalizacion `H(0) = H0`, solo la razon `rho_b0 : rho0` es
observable en el fondo: la reescala conjunta `(rho_b0, rho0) ->
(k rho_b0, k rho0)` deja `H(z)` invariante (test
`TestKnownDegeneracy`). Por eso el ajuste sintetico parametriza

```
f_m = rho_b0 / (rho_b0 + rho0)
```

Cualquier likelihood futura debe muestrear `f_m` (o fijar la
normalizacion) y no las dos amplitudes a la vez.

## Uso con cobaya (opcional, cobaya >= 3.6.2)

```python
from mcmc.cobaya_interface.theory import get_theory_class
MCMCBackground = get_theory_class()

info = {
    "theory": {"mcmc_background": MCMCBackground},
    "params": {
        "H0":      {"prior": {"min": 50, "max": 90},  "ref": 67.4},
        "rho_b0":  {"prior": {"min": 0.05, "max": 1.0}, "ref": 0.30},
        "rho0":    {"prior": {"min": 0.05, "max": 1.0}, "ref": 0.70},
        "z_trans": {"prior": {"min": 0.0, "max": 5.0},  "ref": 1.0},
        "eps":     {"prior": {"min": 0.0, "max": 1.0},  "ref": 0.05},
    },
    # likelihoods BAO/SNe que consuman Hubble / distancias del provider
}
```

Productos publicados: `Hubble`, `comoving_radial_distance`,
`angular_diameter_distance`, `luminosity_distance`; derivados
`Omega_m_eff` y `w_eff_id_0`.

Fase 2 (pendiente): espectros CMB via CLASS/CAMB.

## Ajuste sintetico (Prioridad 2)

```bash
python scripts/run_synthetic_recovery.py --nsteps 1500 --nwalkers 32
```

Genera BAO+SNe+H(z) sinteticos con verdad conocida (semilla fija,
hashes sha256 en `data/synthetic/truth.json`), ajusta MAP + emcee y
escribe `results/synthetic_recovery/recovery_report.json` con pulls por
parametro y el veredicto global. Incluye la comprobacion round-trip del
mapeo entropico `S <-> z` (error maximo ~1e-15).

Resultado de referencia (seed 20260801, 32 walkers x 1500 pasos): los 6
parametros recuperados dentro de 2 sigma.

## Prediccion congelada

```bash
python scripts/freeze_predictions.py --version v1
```

Escribe `results/frozen_prediction_v1.json` (inmutable: el script se
niega a sobreescribir una version existente) con `H(z)`, distancias y
`w_eff_id(z)` fiduciales sellados con sha256, para comparar contra DESI
DR3 / Rubin sin ajuste retrospectivo.

## Procedencia de datos externos

```bash
python scripts/register_provenance.py add <fichero> --source <URL/DOI> --version <ver>
python scripts/register_provenance.py verify
```

Registro versionado en `data/provenance.json`; el test
`tests/test_provenance.py::TestRepoRegistry` verifica los hashes en CI.
Todo producto externo (likelihoods SPT-3G, cadenas corregidas CMB_SPA,
catalogos DESI...) debe registrarse antes de usarse.
