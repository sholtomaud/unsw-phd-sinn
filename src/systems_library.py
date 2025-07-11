# src/systems_library.py

import numpy as np
import logging
import jax.numpy as jnp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SystemModel:
    """
    Base class for all system models.
    Ensures that all models have a consistent interface.
    """
    def get_derivative(self, *args, **kwargs):
        """
        Placeholder for the derivative function.
        This should be implemented by all subclasses.
        """
        raise NotImplementedError("The get_derivative method must be implemented by the subclass.")

class TankSystem(SystemModel):
    """
    Represents Odum's TANK/STORAGE model using pure Python/NumPy.
    This class is the "ground truth" definition of the system's physics.
    """
    def __init__(self, J=2.0, k=0.1, Q0=1.0):
        self.J = J
        self.k = k
        self.Q0 = Q0
        logging.info(f"TankSystem initialized with J={self.J}, k={self.k}, Q0={self.Q0}")

    def get_derivative(self, Q, t=None):
        """
        Defines the system's governing ODE: dQ/dt = J - k*Q.
        This is the single source of truth for the system's dynamics.
        """
        return self.J - self.k * Q

    def solve_analytical(self, t_eval):
        """Calculates the exact, perfect solution at the given time points."""
        Q_steady = self.J / self.k
        return Q_steady + (self.Q0 - Q_steady) * np.exp(-self.k * t_eval)

    def solve_euler(self, t_max, dt):
        """Solves the system using the transparent Euler method."""
        time_steps = np.arange(0, t_max + dt, dt)
        storage_values = np.zeros_like(time_steps)
        storage_values[0] = self.Q0
        
        for i in range(len(time_steps) - 1):
            current_Q = storage_values[i]
            dQ_dt = self.get_derivative(current_Q)
            storage_values[i+1] = current_Q + (dQ_dt * dt)
            
        return time_steps, storage_values


class NoriaSystem(SystemModel):
    """
    Represents a simplified Noria (water wheel) model.
    This class defines the physics of the Noria system.
    """
    def __init__(self, Q_in=1.0, k_q=0.1, k_tau=0.5, k_friction=0.05, I=10.0, h0=0.0, omega0=0.0):
        self.Q_in = Q_in
        self.k_q = k_q
        self.k_tau = k_tau
        self.k_friction = k_friction
        self.I = I
        self.h0 = h0
        self.omega0 = omega0
        logging.info(f"NoriaSystem initialized with Q_in={self.Q_in}, k_q={self.k_q}, k_tau={self.k_tau}, k_friction={self.k_friction}, I={self.I}, h0={self.h0}, omega0={self.omega0}")

    def get_derivative(self, state, t=None):
        """
        Defines the system's governing ODEs: dh/dt and d_omega/dt.
        state = [h, omega]
        """
        h, omega = state
        dh_dt = self.Q_in - self.k_q * h
        d_omega_dt = (self.k_tau * h - self.k_friction * omega) / self.I
        return jnp.array([dh_dt, d_omega_dt])

    def solve_euler(self, t_max, dt):
        """Solves the Noria system using the Euler method."""
        time_steps = np.arange(0, t_max + dt, dt)
        state_values = np.zeros((len(time_steps), 2))
        state_values[0] = np.array([self.h0, self.omega0])

        for i in range(len(time_steps) - 1):
            current_state = state_values[i]
            d_state_dt = self.get_derivative(current_state)
            state_values[i+1] = current_state + (d_state_dt * dt)
            
        return time_steps, state_values
