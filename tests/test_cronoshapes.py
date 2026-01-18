import numpy as np
import pytest
from mcmc.core.cronoshapes import (
    CronosShapeParams, C_of_S, T_of_S, Phi_ten_of_S, N_of_S
)
from mcmc.core.s_grid import create_default_grid
from mcmc.core.background import BackgroundParams, solve_background
from mcmc.core.checks import assert_background_ok


# Sellos ontologicos
S1, S2, S3, S4 = 0.010, 0.100, 1.000, 1.001


def test_C_of_S_decreases_toward_S4():
    """C(S) debe tender a valores menores hacia S4 (expansion se suaviza)."""
    S = np.linspace(0.01, 1.001, 500)
    p = CronosShapeParams()
    C = C_of_S(S, p, S2=S2, S3=S3)

    # C al inicio (regimen rigido) debe ser mayor que al final (inercial)
    assert C[0] > C[-1]
    assert C[0] == pytest.approx(p.C_early, rel=0.1)
    # Nota: con sigmoid width=0.03, en S=1.001 la transicion apenas empieza
    # Por eso solo verificamos que C esta disminuyendo, no que haya alcanzado C_late
    assert C[-1] < p.C_mid, "C debe estar en transicion hacia C_late"


def test_C_of_S_positive():
    """C(S) debe ser positivo en todo el rango."""
    S = np.linspace(0.01, 1.001, 500)
    p = CronosShapeParams()
    C = C_of_S(S, p, S2=S2, S3=S3)

    assert np.all(C > 0)
    assert np.all(np.isfinite(C))


def test_T_of_S_peaks_at_seals():
    """T(S) debe tener picos en los sellos."""
    S = np.linspace(0.005, 1.005, 1000)
    p = CronosShapeParams()
    T = T_of_S(S, p, S1=S1, S2=S2, S3=S3)

    # T base es 1.0, con picos debe ser > 1.0 cerca de sellos
    T_at_S1 = T[np.argmin(np.abs(S - S1))]
    T_at_S2 = T[np.argmin(np.abs(S - S2))]
    T_at_S3 = T[np.argmin(np.abs(S - S3))]

    assert T_at_S1 > p.T0
    assert T_at_S2 > p.T0
    assert T_at_S3 > p.T0


def test_Phi_ten_decreases_from_S1():
    """Phi_ten debe decrecer desde S1 (envolvente exponencial)."""
    S = np.linspace(0.01, 1.0, 500)
    p = CronosShapeParams()
    Phi = Phi_ten_of_S(S, p, S1=S1, S2=S2, S3=S3)

    # La tendencia general debe ser decreciente
    # (aunque haya bultos locales)
    assert Phi[0] > Phi[-1]


def test_N_of_S_positive():
    """N(S) = exp(Phi_ten) debe ser positivo y finito."""
    S = np.linspace(0.01, 1.001, 500)
    p = CronosShapeParams()
    N = N_of_S(S, p, S1=S1, S2=S2, S3=S3)

    assert np.all(N > 0)
    assert np.all(np.isfinite(N))


def test_background_with_cronos_shapes():
    """Integracion del fondo con formas de Cronos refinadas."""
    grid, S = create_default_grid()
    p = BackgroundParams()

    sol = solve_background(
        S, p,
        S1=grid.seals.S1,
        S2=grid.seals.S2,
        S3=grid.seals.S3,
        S4=grid.seals.S4,
    )

    # Debe pasar checks de coherencia
    assert_background_ok(sol)

    # H(z=0) debe ser H0
    assert sol["H"][-1] == pytest.approx(p.H0, rel=1e-6)

    # a(S4) = 1
    assert sol["a"][-1] == pytest.approx(1.0, rel=1e-10)

    # z(S4) = 0
    assert sol["z"][-1] == pytest.approx(0.0, abs=1e-10)


def test_cronos_shapes_monotonic_trends():
    """Test de coherencia: C baja hacia S4, N es positivo."""
    grid, S = create_default_grid()
    sol = solve_background(
        S,
        BackgroundParams(),
        S1=grid.seals.S1,
        S2=grid.seals.S2,
        S3=grid.seals.S3,
        S4=grid.seals.S4,
    )
    assert_background_ok(sol)

    # C(S) debe tender a valores menores hacia S4 (expansion se suaviza)
    assert sol["C"][0] >= sol["C"][-1]

    # N(S) debe ser >0 y finito
    assert np.isfinite(sol["N"]).all()
    assert (sol["N"] > 0).all()

    # Phi_ten debe estar contenido
    assert np.isfinite(sol["Phi_ten"]).all()
