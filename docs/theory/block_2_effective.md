# Bloque II - Cosmologia efectiva (BAO, H(z), SNe) con rho_id(z) refinada

## Friedmann modificado (forma de validacion intermedia)

En lugar de Lambda fija, el modelo usa densidad de vacio cuantico rho_id(z)
con transicion suave:

### Componentes de densidad

```
rho_bar(z) = rho_b0 * (1+z)^3    (barionica/materia)
```

```
rho_id(z) =
  - rho0 * (1+z)^3                   si z > z_trans
  - rho0 * [1 + eps * (z_trans - z)]  si z <= z_trans
```

### Ecuacion de Friedmann efectiva

```
H^2(z) = (8*pi*G/3) * [rho_bar(z) + rho_id(z)]
```

Interpretacion: a bajo z, rho_id ~ casi constante, lo que emula w ~ -1
y produce aceleracion tardia.

## Distancias cosmologicas

### Distancia comoving

```
r(z) = c * integral_0^z dz'/H(z')
```

### Distancia de diametro angular

```
D_A(z) = r(z) / (1+z)
```

### Distancia de volumen (BAO)

```
D_V(z) = [(1+z)^2 * D_A^2(z) * (c*z)/H(z)]^(1/3)
```

### Distancia de luminosidad (SNe)

```
d_L(z) = (1+z) * r(z)
```

### Modulo de distancia

```
mu(z) = 5 * log10(d_L/Mpc) + 25
```

(+ parametro M si se marginaliza)

## Observables y datos

### BAO
- BOSS DR12: D_V(z)/r_d a z = 0.38, 0.51, 0.61
- eBOSS: extensiones a z > 1

### H(z) - Cosmic chronometers
- Compilacion de ~30 puntos en z in [0, 2]
- Incertidumbres tipicas 5-10 km/s/Mpc

### SNe Ia
- Pantheon: ~1000 SNe en z in [0.01, 2.3]
- Pantheon+: actualizacion con mas SNe cercanas

## Ajuste estadistico

### Chi-cuadrado individual

```
chi2_X = sum_i [(X_obs,i - X_mod,i) / sigma_i]^2
```

### Chi-cuadrado total

```
chi2_total = chi2_BAO + chi2_Hz + chi2_SNe
```

### Criterios de informacion

```
AIC = 2*k + chi2_min
BIC = k*ln(N) + chi2_min
```

Donde k = numero de parametros, N = numero de datos.

## Parametros del modelo

Para validacion intermedia:
- H0: constante de Hubble [50, 90] km/s/Mpc
- rho_b0: densidad barionica normalizada [0.1, 0.5]
- rho0: amplitud rho_id [0.0, 2.0]
- z_trans: redshift de transicion [0.5, 3.0]
- eps: pendiente post-transicion [0.0, 0.5]

## Implementacion

- rho_id(z) transicion: `src/mcmc/channels/rho_id_refined.py`
- H(z) efectivo: `src/mcmc/core/friedmann_effective.py`
- Distancias: `src/mcmc/observables/distances.py`, `bao_distances.py`
- Likelihood global: `src/mcmc/observables/likelihoods.py`
