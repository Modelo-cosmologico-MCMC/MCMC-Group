#!/usr/bin/env python3
"""Block 5: Qubit Tensorial MCMC Simulations and Visualizations.

Generates PNG plots for:
1. Qudit Hilbert space and S mapping
2. Tensorial operators
3. MCMC Hamiltonian spectrum
4. Time evolution
5. Quantum gates
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Block 5 imports
from mcmc.blocks.block5.config import HilbertSpaceParams, HamiltonianParams
from mcmc.blocks.block5.hilbert_space import (
    QuditState,
    QuditBasis,
    create_vacuum,
    create_S_eigenstate,
)
from mcmc.blocks.block5.operators import (
    creation_operator,
    annihilation_operator,
    number_operator,
    S_operator,
)
from mcmc.blocks.block5.hamiltonian import (
    MCMCHamiltonian,
    kinetic_term,
    potential_term,
    time_evolution,
)
from mcmc.blocks.block5.gates import hadamard_gate, phase_gate, rotation_gate
from mcmc.core.ontology import S_0, S_GEOM

# Output directory
OUTDIR = Path("reports/figures/blocks")
OUTDIR.mkdir(parents=True, exist_ok=True)


def plot_hilbert_space():
    """Plot Qudit Hilbert space and S mapping."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    params = HilbertSpaceParams(d=20, S_min=0.0, S_max=100.0)
    basis = QuditBasis(params)

    # Level to S mapping
    ax = axes[0, 0]
    levels = np.arange(params.d)
    S_values = [basis.level_to_S(k) for k in levels]

    ax.bar(levels, S_values, color='steelblue', alpha=0.7)
    ax.axhline(S_GEOM, color='r', linestyle='--', label=f'S_GEOM = {S_GEOM}')
    ax.axhline(S_0, color='g', linestyle='--', label=f'S_0 = {S_0:.1f}')
    ax.set_xlabel('Qudit Level k')
    ax.set_ylabel('S value')
    ax.set_title('Level-to-S Mapping')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Vacuum state
    ax = axes[0, 1]
    vacuum = create_vacuum(params)
    probs = np.abs(vacuum.amplitudes) ** 2

    ax.bar(levels, probs, color='purple', alpha=0.7)
    ax.set_xlabel('Qudit Level k')
    ax.set_ylabel('Probability')
    ax.set_title(f'Vacuum State |0> (S = {vacuum.S_value():.2f})')
    ax.grid(True, alpha=0.3)

    # S eigenstate
    ax = axes[1, 0]
    state_50 = create_S_eigenstate(S=50.0, params=params)
    probs_50 = np.abs(state_50.amplitudes) ** 2

    ax.bar(levels, probs_50, color='green', alpha=0.7)
    ax.set_xlabel('Qudit Level k')
    ax.set_ylabel('Probability')
    ax.set_title(f'S Eigenstate (S = 50, measured = {state_50.S_value():.2f})')
    ax.grid(True, alpha=0.3)

    # Gaussian superposition
    ax = axes[1, 1]
    coeffs = np.exp(-0.5 * (levels - 10) ** 2 / 9)
    coeffs = coeffs / np.linalg.norm(coeffs)
    superpos = QuditState(coeffs, params.d, params)
    probs_sup = np.abs(superpos.amplitudes) ** 2

    ax.bar(levels, probs_sup, color='orange', alpha=0.7)
    ax.set_xlabel('Qudit Level k')
    ax.set_ylabel('Probability')
    ax.set_title(f'Gaussian Superposition (S = {superpos.S_value():.2f})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTDIR / "block5_01_hilbert_space.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTDIR / 'block5_01_hilbert_space.png'}")


def plot_operators():
    """Plot tensorial operators."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    params = HilbertSpaceParams(d=10, S_min=0.0, S_max=100.0)

    # Creation operator
    ax = axes[0, 0]
    a_dag = creation_operator(params)
    im = ax.imshow(np.abs(a_dag.matrix), cmap='Blues')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(r'$|a^\dagger|$ (Creation Operator)')
    plt.colorbar(im, ax=ax)

    # Annihilation operator
    ax = axes[0, 1]
    a = annihilation_operator(params)
    im = ax.imshow(np.abs(a.matrix), cmap='Reds')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title('|a| (Annihilation Operator)')
    plt.colorbar(im, ax=ax)

    # Number operator
    ax = axes[0, 2]
    n = number_operator(params)
    im = ax.imshow(np.real(n.matrix), cmap='viridis')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title('n (Number Operator)')
    plt.colorbar(im, ax=ax)

    # S operator
    ax = axes[1, 0]
    S_op = S_operator(params)
    im = ax.imshow(np.real(S_op.matrix), cmap='plasma')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(r'$\hat{S}$ (Entropic Operator)')
    plt.colorbar(im, ax=ax)

    # Commutator [a, a^dag]
    ax = axes[1, 1]
    commutator = a.matrix @ a_dag.matrix - a_dag.matrix @ a.matrix
    im = ax.imshow(np.real(commutator), cmap='RdBu', vmin=-1.5, vmax=1.5)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(r'$[a, a^\dagger]$ (Commutator)')
    plt.colorbar(im, ax=ax)

    # Eigenvalues of S operator
    ax = axes[1, 2]
    eigenvalues = np.linalg.eigvalsh(S_op.matrix)
    ax.bar(range(len(eigenvalues)), eigenvalues, color='purple', alpha=0.7)
    ax.set_xlabel('Eigenstate index')
    ax.set_ylabel('S eigenvalue')
    ax.set_title(r'Eigenvalues of $\hat{S}$')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTDIR / "block5_02_operators.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTDIR / 'block5_02_operators.png'}")


def plot_hamiltonian():
    """Plot MCMC Hamiltonian spectrum."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    params = HilbertSpaceParams(d=15, S_min=0.0, S_max=100.0)
    h_params = HamiltonianParams(omega=1.0, V_amplitude=0.5)
    H = MCMCHamiltonian(params, h_params)

    # Hamiltonian matrix
    ax = axes[0, 0]
    im = ax.imshow(np.real(H.matrix), cmap='RdBu')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title('MCMC Hamiltonian Matrix')
    plt.colorbar(im, ax=ax)

    # Energy spectrum
    ax = axes[0, 1]
    eigenvalues = H.eigenvalues()
    ax.bar(range(len(eigenvalues)), eigenvalues, color='steelblue', alpha=0.7)
    ax.set_xlabel('Level n')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Spectrum')
    ax.grid(True, alpha=0.3)

    # Kinetic term
    ax = axes[1, 0]
    H_kin = kinetic_term(params, h_params)
    im = ax.imshow(np.real(H_kin.matrix), cmap='Blues')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(r'$H_{kin} = \omega \hat{n}$')
    plt.colorbar(im, ax=ax)

    # Potential term
    ax = axes[1, 1]
    H_pot = potential_term(params, h_params)
    im = ax.imshow(np.real(H_pot.matrix), cmap='Greens')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title('V(S) (Potential Term)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(OUTDIR / "block5_03_hamiltonian.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTDIR / 'block5_03_hamiltonian.png'}")


def plot_time_evolution():
    """Plot time evolution."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    params = HilbertSpaceParams(d=15, S_min=0.0, S_max=100.0)
    h_params = HamiltonianParams(omega=1.0, V_amplitude=0.3)
    H = MCMCHamiltonian(params, h_params)

    # Initial state
    initial = create_S_eigenstate(S=30.0, params=params)

    # Unitary evolution
    ax = axes[0, 0]
    times = np.linspace(0, 5, 50)
    S_values_unitary = []

    for t in times:
        evolved = time_evolution(initial, H, t, n_steps=10)
        S_values_unitary.append(evolved.S_value())

    ax.plot(times, S_values_unitary, 'b-', linewidth=2)
    ax.axhline(initial.S_value(), color='gray', linestyle='--', alpha=0.5,
               label='Initial S')
    ax.set_xlabel('Time')
    ax.set_ylabel('S')
    ax.set_title('Unitary Evolution: <S>(t)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Energy expectation
    ax = axes[0, 1]
    E_values = []
    for t in times:
        evolved = time_evolution(initial, H, t, n_steps=10)
        E_values.append(H.expect(evolved))

    ax.plot(times, E_values, 'g-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Expectation <E>(t)')
    ax.grid(True, alpha=0.3)

    # Probability evolution
    ax = axes[1, 0]
    levels = np.arange(params.d)
    prob_history = []
    time_points = [0, 1, 2, 3, 4, 5]

    for t in time_points:
        evolved = time_evolution(initial, H, t, n_steps=10)
        prob_history.append(np.abs(evolved.amplitudes) ** 2)

    for i, (t, probs) in enumerate(zip(time_points, prob_history)):
        ax.plot(levels, probs + i * 0.15, 'o-', linewidth=1, markersize=3,
                label=f't={t}')

    ax.set_xlabel('Level k')
    ax.set_ylabel('P(k) + offset')
    ax.set_title('Probability Distribution P(k, t)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ground state
    ax = axes[1, 1]
    E_0, psi_0 = H.ground_state()
    probs_gs = np.abs(psi_0.amplitudes) ** 2

    ax.bar(levels, probs_gs, color='green', alpha=0.7)
    ax.set_xlabel('Level k')
    ax.set_ylabel('Probability')
    ax.set_title(f'Ground State (E_0 = {E_0:.3f}, S = {psi_0.S_value():.2f})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTDIR / "block5_04_time_evolution.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTDIR / 'block5_04_time_evolution.png'}")


def plot_quantum_gates():
    """Plot quantum gates."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    params = HilbertSpaceParams(d=8, S_min=0.0, S_max=100.0)

    # Hadamard gate
    ax = axes[0, 0]
    H_gate = hadamard_gate(params)
    im = ax.imshow(np.abs(H_gate.matrix), cmap='Blues')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(f'|H| (Hadamard-like, d={params.d})')
    plt.colorbar(im, ax=ax)

    # Phase gate
    ax = axes[0, 1]
    P_gate = phase_gate(np.pi / 4, params)
    im = ax.imshow(np.angle(P_gate.matrix), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(r'Phase Gate $P(\pi/4)$')
    plt.colorbar(im, ax=ax, label='Phase')

    # Rotation gate
    ax = axes[0, 2]
    R_gate = rotation_gate(np.pi / 3, axis='z', params=params)
    im = ax.imshow(np.abs(R_gate.matrix), cmap='Greens')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(r'$|R_z(\pi/3)|$ (Rotation)')
    plt.colorbar(im, ax=ax)

    # Gate action on vacuum
    ax = axes[1, 0]
    vacuum = create_vacuum(params)
    # Apply gate via matrix multiplication
    after_H_amps = H_gate.matrix @ vacuum.amplitudes
    after_H = QuditState(after_H_amps, params.d, params)

    levels = np.arange(params.d)
    ax.bar(levels - 0.2, np.abs(vacuum.amplitudes) ** 2, width=0.4,
           label='Before H', alpha=0.7)
    ax.bar(levels + 0.2, np.abs(after_H.amplitudes) ** 2, width=0.4,
           label='After H', alpha=0.7)
    ax.set_xlabel('Level k')
    ax.set_ylabel('Probability')
    ax.set_title('Hadamard Action on |0>')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Unitarity check
    ax = axes[1, 1]
    gates = [
        ('H', hadamard_gate(params)),
        ('P', phase_gate(np.pi / 4, params)),
        ('Rz', rotation_gate(np.pi / 3, 'z', params)),
        ('Rx', rotation_gate(np.pi / 3, 'x', params)),
    ]

    unitary_errors = []
    gate_names = []
    for name, gate in gates:
        identity = np.eye(params.d)
        U_Udag = gate.matrix @ gate.matrix.conj().T
        error = np.max(np.abs(U_Udag - identity))
        unitary_errors.append(error)
        gate_names.append(name)

    ax.bar(gate_names, unitary_errors, color='steelblue', alpha=0.7)
    ax.axhline(1e-10, color='r', linestyle='--', label='Tolerance')
    ax.set_ylabel('Unitarity Error')
    ax.set_title('Gate Unitarity Check')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Circuit composition (apply gates via matrix multiplication)
    ax = axes[1, 2]
    state = create_vacuum(params)
    states = [state]
    labels = ['|0>']

    # Apply H gate
    new_amps = H_gate.matrix @ state.amplitudes
    state = QuditState(new_amps, params.d, params)
    states.append(state)
    labels.append('H|0>')

    # Apply P gate
    new_amps = P_gate.matrix @ state.amplitudes
    state = QuditState(new_amps, params.d, params)
    states.append(state)
    labels.append('PH|0>')

    # Apply Rz gate
    Rz_gate = rotation_gate(np.pi / 3, 'z', params)
    new_amps = Rz_gate.matrix @ state.amplitudes
    state = QuditState(new_amps, params.d, params)
    states.append(state)
    labels.append('RzPH|0>')

    for i, (s, label) in enumerate(zip(states, labels)):
        ax.plot(levels, np.abs(s.amplitudes) ** 2 + i * 0.3,
                'o-', linewidth=1, markersize=4, label=label)

    ax.set_xlabel('Level k')
    ax.set_ylabel('P(k) + offset')
    ax.set_title('Gate Sequence: H -> P -> Rz')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTDIR / "block5_05_quantum_gates.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTDIR / 'block5_05_quantum_gates.png'}")


def main():
    """Run all Block 5 visualizations."""
    print("=" * 60)
    print("Block 5: Qubit Tensorial MCMC Simulations")
    print("=" * 60)

    plot_hilbert_space()
    plot_operators()
    plot_hamiltonian()
    plot_time_evolution()
    plot_quantum_gates()

    print("\nBlock 5 visualizations complete!")


if __name__ == "__main__":
    main()
