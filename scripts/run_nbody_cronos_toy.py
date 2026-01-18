from __future__ import annotations

import argparse
import numpy as np

from mcmc.nbody.kdk import State, integrate
from mcmc.nbody.crono_step import CronosStepParams, delta_t_cronos, global_timestep
from mcmc.nbody.poisson import PoissonParams, make_acceleration_fn, estimate_density


def generate_initial_conditions(N: int, box_size: float, seed: int = 42) -> State:
    """
    Genera condiciones iniciales aleatorias en una caja.

    Args:
        N: Numero de particulas
        box_size: Tamano de la caja
        seed: Semilla aleatoria

    Returns:
        Estado inicial
    """
    rng = np.random.default_rng(seed)

    # Posiciones uniformes en la caja
    x = rng.uniform(0, box_size, size=(N, 3))

    # Velocidades pequenas (dispersion termica)
    v = rng.normal(0, 0.1, size=(N, 3))

    # Masas iguales
    m = np.ones(N)

    return State(x=x, v=v, m=m)


def run_with_cronos_timestep(
    state: State,
    acc_fn,
    n_steps: int,
    cronos_params: CronosStepParams,
    a_scale: float = 1.0,
    h_density: float = 0.1
) -> list[State]:
    """
    Integra usando timestep Cronos adaptativo.

    Args:
        state: Estado inicial
        acc_fn: Funcion de aceleracion
        n_steps: Numero de pasos
        cronos_params: Parametros de Cronos
        a_scale: Factor de escala cosmico
        h_density: Radio para estimacion de densidad

    Returns:
        Lista de estados
    """
    from mcmc.nbody.kdk import kdk_step

    states = [state]
    current = state

    for _ in range(n_steps):
        # Calcular aceleraciones
        acc = acc_fn(current.x)
        acc_norm = np.linalg.norm(acc, axis=1)

        # Estimar densidad local
        rho = estimate_density(current.x, current.m, h_density)

        # Calcular timestep Cronos
        dt_particles = delta_t_cronos(acc_norm, a_scale, rho, cronos_params)
        dt = global_timestep(dt_particles)

        # Integrar
        current = kdk_step(current, dt, acc_fn)
        states.append(current)

    return states


def main() -> None:
    """
    Ejecuta simulacion N-body toy con timestep Cronos.
    """
    ap = argparse.ArgumentParser(
        description="Simulacion N-body toy con timestep Cronos"
    )
    ap.add_argument("--N", type=int, default=100,
                    help="Numero de particulas")
    ap.add_argument("--box", type=float, default=1.0,
                    help="Tamano de la caja")
    ap.add_argument("--steps", type=int, default=100,
                    help="Numero de pasos")
    ap.add_argument("--eta", type=float, default=0.02,
                    help="Parametro eta de Cronos")
    ap.add_argument("--seed", type=int, default=42,
                    help="Semilla aleatoria")
    ap.add_argument("--use-cronos", action="store_true",
                    help="Usar timestep Cronos (default: dt fijo)")
    ap.add_argument("--dt", type=float, default=0.01,
                    help="Paso de tiempo fijo (si no usa Cronos)")
    args = ap.parse_args()

    print(f"=== N-body Cronos Toy Model ===")
    print(f"N = {args.N}, box = {args.box}, steps = {args.steps}")
    print(f"Timestep: {'Cronos' if args.use_cronos else 'fijo'}")

    # Generar condiciones iniciales
    state0 = generate_initial_conditions(args.N, args.box, args.seed)
    print(f"Energia cinetica inicial: {state0.kinetic_energy():.4f}")

    # Configurar aceleracion
    poisson_params = PoissonParams(G=1.0, softening=0.01, box_size=args.box)
    acc_fn = make_acceleration_fn(state0.m, poisson_params)

    # Integrar
    if args.use_cronos:
        cronos_params = CronosStepParams(eta=args.eta, rho_c=1.0, alpha=1.0)
        states = run_with_cronos_timestep(
            state0, acc_fn, args.steps, cronos_params,
            a_scale=1.0, h_density=0.1
        )
    else:
        states = integrate(state0, acc_fn, args.dt, args.steps, save_every=10)

    # Resultados
    final_state = states[-1]
    print(f"Energia cinetica final: {final_state.kinetic_energy():.4f}")
    print(f"Estados guardados: {len(states)}")
    print("OK: Simulacion completada")


if __name__ == "__main__":
    main()
