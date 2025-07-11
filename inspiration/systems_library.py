# systems_library.py

import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TankSystem:
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

if __name__ == "__main__":
    logging.info("--- Verifying the TankSystem Definition in systems_library.py ---")
    tank = TankSystem()
    t_plot = np.linspace(0, 50, 500)
    Q_analytical = tank.solve_analytical(t_plot)
    t_euler, Q_euler = tank.solve_euler(t_max=50, dt=0.1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, Q_analytical, 'k-', lw=3, label='Analytical Solution')
    plt.plot(t_euler, Q_euler, 'b--', label='Euler Simulation (dt=0.1)')
    plt.title("Verification of Solvers in systems_library.py")
    plt.xlabel("Time (t)")
    plt.ylabel("Storage (Q)")
    plt.legend()
    plt.grid(True)
    plt.show()