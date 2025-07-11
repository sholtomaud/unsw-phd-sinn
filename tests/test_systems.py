# tests/test_systems.py

import numpy as np
import pytest
from src.systems_library import TankSystem

@pytest.fixture
def tank_model():
    """Provides a default TankSystem instance for testing."""
    return TankSystem(J=2.0, k=0.1, Q0=1.0)

def test_get_derivative(tank_model):
    """Tests the dQ/dt calculation at a specific point."""
    # At Q=10, dQ/dt should be J - k*Q = 2.0 - 0.1*10 = 1.0
    assert tank_model.get_derivative(Q=10) == pytest.approx(1.0)
    # At Q=20 (steady state), dQ/dt should be J - k*Q = 2.0 - 0.1*20 = 0.0
    assert tank_model.get_derivative(Q=20) == pytest.approx(0.0)

def test_euler_approximates_analytical(tank_model):
    """
    Tests that the Euler method closely approximates the analytical solution
    with a reasonably small time step (dt).
    """
    t_max = 50
    dt = 0.1
    t_euler, Q_euler = tank_model.solve_euler(t_max, dt)
    Q_analytical = tank_model.solve_analytical(t_euler)
    
    # Check if the mean squared error is below a small threshold
    mse = np.mean((Q_euler - Q_analytical)**2)
    assert mse < 1e-3
