# ==============================================================================
# main_app.py
#
# Main application script for training and validating a PINN for a
# specific system model. This script brings all the modules together.
# ==============================================================================

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.struct import field
import matplotlib.pyplot as plt
import numpy as np
import logging

# --- Import our modular components ---
from pinn_framework import PINN_Framework
from systems_library import TankSystem

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# SECTION 1: PROBLEM-SPECIFIC DEFINITIONS (The "JAX-ify" part)
# This section translates the pure physics model into JAX/Flax components.
# In a larger project, this could be its own `pinn_components.py` file.
# ==============================================================================

# --- The PINN Architecture for this problem ---
class TankPINN(nn.Module):
    """A simple PINN to learn the function Q(t)."""
    features: list[int] = field(default_factory=lambda: [32, 32, 32, 1])

    @nn.compact
    def __call__(self, t):
        # The model only needs time 't' as input to predict the state 'Q'.
        x = t.reshape(-1, 1)
        for i, feat in enumerate(self.features):
            x = nn.Dense(features=feat)(x)
            if i < len(self.features) - 1:
                x = nn.tanh(x)
        # Use softplus to ensure the output Q is always positive.
        return nn.softplus(x).squeeze()

# --- The Physics-Informed Loss Function for this problem ---
def tank_loss_fn(params, model, t_coll, t_initial, Q_initial, system: TankSystem):
    """
    Calculates loss based on the TANK ODE by calling the system's get_derivative method.
    This function is passed to the generic framework.

    Args:
        params: The current parameters of the neural network.
        model: The Flax model architecture.
        t_coll: A batch of random time points for checking the physics residual.
        t_initial: The time of the initial condition (e.g., 0.0).
        Q_initial: The value of the initial condition (e.g., 1.0).
        system: The pure Python system object containing physical constants (J, k).
    """
    # Define a function for Q(t) that uses the current model parameters
    Q_fn = lambda t_var: model.apply({'params': params}, t_var)
    # Get the derivative function dQ/dt using jax.grad
    dQ_dt_fn = jax.grad(Q_fn)

    # --- Physics Loss ---
    # Evaluate the model and its derivative at all collocation points
    Q_pred_coll = jax.vmap(Q_fn)(t_coll)
    dQ_dt_pred_coll = jax.vmap(dQ_dt_fn)(t_coll)
    
    # Get the "right-hand side" of the ODE from our single source of truth
    physics_rhs = jax.vmap(system.get_derivative)(Q_pred_coll)
    
    # The residual is the difference between the two sides of the ODE
    residual_physics = dQ_dt_pred_coll - physics_rhs
    loss_physics = jnp.mean(residual_physics**2)

    # --- Initial Condition Loss ---
    Q_pred_initial = Q_fn(t_initial)
    loss_initial = (Q_pred_initial - Q_initial)**2
    
    # Return the weighted sum of the losses
    return loss_physics + 100.0 * loss_initial

# --- The Data Generator for this problem ---
def tank_data_generator(Q0, collocation_size=200, t_max=50.0):
    """Yields batches of random collocation points and initial conditions."""
    key = jax.random.PRNGKey(0)
    t_initial = 0.0
    while True:
        key, subkey = jax.random.split(key)
        t_coll = jax.random.uniform(subkey, (collocation_size,)) * t_max
        # Yield a tuple of the DYNAMIC arguments for the loss function
        yield (t_coll, t_initial, Q0)

# ==============================================================================
# SECTION 2: MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    # 1. DEFINE THE SPECIFIC SYSTEM TO SOLVE
    # Instantiate the pure physics model from our library
    system_instance = TankSystem(J=2.0, k=0.1, Q0=1.0)
    
    # 2. INSTANTIATE THE PINN ARCHITECTURE
    pinn_model_arch = TankPINN()
    dummy_t = jnp.ones((1,))
    
    # 3. INSTANTIATE THE GENERIC FRAMEWORK
    pinn_solver = PINN_Framework(
        model=pinn_model_arch,
        loss_fn=tank_loss_fn,
        dummy_inputs=(dummy_t,)
    )
    
    # 4. TRAIN THE MODEL
    data_gen = tank_data_generator(Q0=system_instance.Q0)
    
    # Pass the system object as a static argument to the train method
    loss_history = pinn_solver.train(
        training_data_generator=data_gen,
        num_epochs=75000,
        static_loss_args={'system': system_instance}
    )
    
    # 5. SAVE THE TRAINED MODEL SNAPSHOT
    pinn_solver.save_snapshot(directory='./tank_model_snapshot', step=20000)
    
    # 6. VALIDATE AND VISUALIZE
    # Get ground truth from our verified systems library
    t_plot = np.linspace(0, 50, 500)
    Q_analytical = system_instance.solve_analytical(t_plot)
    
    # Get prediction from the trained PINN
    predictor = pinn_solver.get_predictor()
    Q_pinn = jax.vmap(predictor)(t_plot)

    # Plot the final comparison
    plt.figure(figsize=(12, 8))
    plt.plot(t_plot, Q_analytical, 'k-', linewidth=3, label='Ground Truth (from systems_library)')
    plt.plot(t_plot, Q_pinn, 'r--', linewidth=2, label='PINN Prediction')
    plt.scatter([0.0], [system_instance.Q0], color='blue', zorder=5, s=100, label='Initial Condition')
    plt.xlabel('Time (t)')
    plt.ylabel('Storage (Q)')
    plt.title("Verification: PINN vs. Ground Truth System Model")
    plt.legend()
    plt.grid(True)
    plt.show()