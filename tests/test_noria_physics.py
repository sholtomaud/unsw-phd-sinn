import numpy as np
import pytest
from src.systems_library import NoriaSystem

@pytest.fixture
def noria_model():
    """Provides a default NoriaSystem instance for testing."""
    return NoriaSystem(Q_in=1.0, k_q=0.1, k_tau=0.5, k_friction=0.05, I=10.0, h0=0.0, omega0=0.0)

def test_noria_get_derivative(noria_model):
    """
    Tests the dh/dt and d_omega/dt calculations at a specific point.
    State = [h, omega]
    """
    # Test at initial state [0, 0]
    # dh/dt = Q_in - k_q * h = 1.0 - 0.1 * 0 = 1.0
    # d_omega/dt = (k_tau * h - k_friction * omega) / I = (0.5 * 0 - 0.05 * 0) / 10 = 0.0
    expected_derivative_initial = np.array([1.0, 0.0])
    assert np.allclose(noria_model.get_derivative(np.array([0.0, 0.0])), expected_derivative_initial)

    # Test at a different state [10, 5]
    # dh/dt = 1.0 - 0.1 * 10 = 0.0
    # d_omega/dt = (0.5 * 10 - 0.05 * 5) / 10 = (5.0 - 0.25) / 10 = 4.75 / 10 = 0.475
    expected_derivative_mid = np.array([0.0, 0.475])
    assert np.allclose(noria_model.get_derivative(np.array([10.0, 5.0])), expected_derivative_mid)

def test_noria_euler_solution(noria_model):
    """
    Tests that the Euler method for NoriaSystem produces reasonable results.
    Since there's no analytical solution, we check for basic properties:
    - h and omega remain non-negative.
    - h and omega change over time (not static unless at equilibrium).
    """
    t_max = 100
    dt = 0.1
    time_steps, state_values = noria_model.solve_euler(t_max, dt)

    h_values = state_values[:, 0]
    omega_values = state_values[:, 1]

    # Check if h and omega remain non-negative
    assert np.all(h_values >= 0)
    assert np.all(omega_values >= 0)

    # Check if values change over time (unless at equilibrium, which is not the case here initially)
    assert not np.allclose(h_values[0], h_values[-1])
    assert not np.allclose(omega_values[0], omega_values[-1])

    # Check for expected general trend (h should increase initially, omega should increase)
    # This is a very basic check and might need adjustment based on specific parameters
    assert h_values[-1] > h_values[0] # h should increase towards equilibrium
    assert omega_values[-1] > omega_values[0] # omega should increase as h increases
