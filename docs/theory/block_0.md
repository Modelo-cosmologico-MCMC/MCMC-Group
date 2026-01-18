# Bloque 0: Estado Pre-Geometrico

## Descripcion

El Bloque 0 describe el estado pre-geometrico del universo antes de que la
geometria clasica emerja. Este bloque genera las condiciones iniciales para
el Bloque I.

## Parametros

- `eps`: Imperfeccion primordial (tipicamente 0.01)
- `phi0`: Campo tensional inicial
- `k0`: Rigidez pre-geometrica
- `S_start`: Punto de entrada al Bloque I (S = 0.010)

## Contrato de Salida

El Bloque 0 produce un JSON con:

```json
{
  "Mp_pre": 0.99,
  "Ep_pre": 0.01,
  "phi_pre": 0.0,
  "k_pre": 1.0,
  "S_start": 0.010
}
```

## Implementacion

- `src/mcmc/pregeom/s0_state.py`: Logica de calculo
- `scripts/run_pregeom_export.py`: Script de exportacion

## Uso

```bash
python scripts/run_pregeom_export.py --eps 0.01 --out results/initial_conditions.json
```
