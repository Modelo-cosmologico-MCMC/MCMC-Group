# Bloque I - Nucleo ontologico computacional (S -> a,z,t_rel,H)

## Rejilla entropica y sellos

- Paso de cuantizacion: dS = 1e-3
- Intervalo: S in [0.010, 1.001]
- Numero de puntos: N_S = 991
- Sellos ontologicos:
  - S1 = 0.010 (inicio post-geometrico)
  - S2 = 0.100 (transicion temprana)
  - S3 = 1.000 (transicion tardia)
  - S4 = 1.001 (hoy, normalizacion)

## Ley de Cronos efectiva

Se definen cuatro funciones fundamentales:

### C(S) - Funcion de expansion
Controla d(ln a)/dS. Representa la respuesta de expansion al drenaje Mp/Ep.

Estructura por regimenes:
- Regimen rigido (S < S2): C ~ 2.2
- Regimen intermedio (S2 <= S < S3): C ~ 1.7
- Regimen casi inercial (S >= S3): C -> 1.0

Transiciones suaves via sigmoides en S2 y S3.

### T(S) - Escala temporal emergente
Cadencia base de cronificacion. Presenta picos controlados en los sellos
(cronificacion concentrada en colapsos).

```
T(S) = T0 * (1 + t1*gauss(S,S1) + t2*gauss(S,S2) + t3*gauss(S,S3))
```

### Phi_ten(S) - Campo de Adrian tensional
Campo tensional en fase relativa. Combina:
- Envolvente exponencial decreciente desde S1
- Bultos locales gaussianos en S1, S2, S3

```
Phi_ten(S) = phi_env * exp(-lambda*(S-S1)) + sum_i phi_i * gauss(S, S_i)
```

### N(S) - Lapse entropico
```
N(S) = exp(Phi_ten(S))
```

Modula el tiempo emergente. Intuicion: no hay tiempo ni expansion sin
tension masa-espacio.

## Ecuaciones integradas

Sobre la rejilla discreta S:

```
(1) d(ln a)/dS = C(S)
(2) d(t_rel)/dS = T(S) * N(S)
```

Integracion hacia atras desde S4.

## Normalizacion en S4

- a(S4) = 1
- t_rel(S4) = 0
- H(S4) = H0 = 67.4 km/s/Mpc

## Definiciones derivadas

```
z(S) = a(S4)/a(S) - 1 = 1/a(S) - 1
H(S) = H0 * C(S)/C(S4)
```

La segunda ecuacion es el nucleo computacional: H hereda la estructura
de C(S), normalizada a H0 en S4.

## Implementacion

- Rejilla S: `src/mcmc/core/s_grid.py`
- Formas Cronos (C,T,Phi_ten,N): `src/mcmc/core/cronoshapes.py`
- Integracion (a,z,t_rel,H): `src/mcmc/core/background.py`
- Checks de coherencia: `src/mcmc/core/checks.py`
