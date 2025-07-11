# noria_optimization.py

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.struct import field
import optax
import logging
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# --- Import our modular components ---
from src.systems_library import NoriaSystem
from src.pinn_framework import PINN_Framework
from src.pinn_builder import build_pinn_model # Import the new builder

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# SECTION 1: OPTIMIZATION PROBLEM DEFINITION
# ==============================================================================

# Define the objective function to be optimized
def objective_function(design_params, pinn_solver, t_eval, system_constants):
    """
    Objective function for Noria design optimization.
    We want to maximize water output (h) at the end of the simulation.
    Since optimizers typically minimize, we will minimize the negative of h.

    Args:
        design_params (dict): Dictionary of design parameters (e.g., 'Q_in', 'k_q').
        pinn_solver (PINN_Framework): The trained PINN solver instance.
        t_eval (jnp.ndarray): Time points for evaluation.
        system_constants (dict): Other system parameters that are not being optimized.

    Returns:
        jnp.ndarray: The negative of the water height at the final time point.
    """
    # Create a NoriaSystem instance with the current design parameters
    # and other fixed system constants.
    current_system = NoriaSystem(
        Q_in=design_params['Q_in'],
        k_q=design_params['k_q'],
        k_tau=system_constants['k_tau'],
        k_friction=system_constants['k_friction'],
        I=system_constants['I'],
        h0=system_constants['h0'],
        omega0=system_constants['omega0']
    )

    # The PINN was trained to predict h and omega based on time only.
    # For optimization, we need to ensure the PINN can somehow be influenced by design parameters.
    # This is a simplified approach: we are assuming the PINN is general enough
    # to implicitly capture the effect of these parameters through its training on a range of data.
    # A more robust approach would involve training a PINN that takes design parameters as inputs.
    
    # For now, we will use the PINN to predict h and omega over time.
    # The PINN's internal parameters are fixed (from training).
    predictor = pinn_solver.get_predictor()
    predictions = jax.vmap(predictor)(t_eval)
    h_pred = predictions[:, 0]

    # We want to maximize h at the final time point, so minimize -h_final
    return -h_pred[-1]

# ==============================================================================
# SECTION 2: MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Noria design optimization.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of optimization epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer.')
    parser.add_argument('--snapshot_dir', type=str, default='./snapshots/noria_model_snapshot', help='Directory of the pre-trained Noria model snapshot.')
    args = parser.parse_args()

    # Load the pre-trained Noria PINN model
    pinn_model_arch = build_pinn_model(output_dim=2) # Noria has 2 outputs (h, omega)
    dummy_t = jnp.ones((1,))
    pinn_solver = PINN_Framework.load_snapshot(model=pinn_model_arch, dummy_inputs=(dummy_t,), checkpoint_dir=args.snapshot_dir)
    
    # Define initial design parameters (these will be optimized)
    initial_design_params = {
        'Q_in': jnp.array(1.0),
        'k_q': jnp.array(0.1)
    }

    # Define fixed system constants (these are not optimized)
    fixed_system_constants = {
        'k_tau': 0.5,
        'k_friction': 0.05,
        'I': 10.0,
        'h0': 1.0,
        'omega0': 0.0
    }

    # Time points for evaluation
    t_eval = jnp.linspace(0, 50, 501) # Must match the training time range and points

    # Initialize optimizer
    optimizer = optax.adam(learning_rate=args.learning_rate)
    opt_state = optimizer.init(initial_design_params)

    # Define the gradient function
    grad_objective_fn = jax.value_and_grad(objective_function, argnums=0)

    logging.info("Starting Noria design optimization...")
    design_history = []
    objective_history = []

    # Optimization loop
    current_design_params = initial_design_params
    for epoch in range(args.epochs):
        objective_val, grads = grad_objective_fn(current_design_params, pinn_solver, t_eval, fixed_system_constants)
        updates, opt_state = optimizer.update(grads, opt_state, current_design_params)
        current_design_params = optax.apply_updates(current_design_params, updates)

        design_history.append({k: v.item() for k, v in current_design_params.items()})
        objective_history.append(objective_val.item())

        if (epoch + 1) % 100 == 0:
            logging.info(f"Epoch {epoch+1:5d} | Objective: {objective_val:.6f} | Q_in: {current_design_params['Q_in']:.4f} | k_q: {current_design_params['k_q']:.4f}")

    logging.info("Optimization finished.")

    # Save optimization history to CSV
    df_history = pd.DataFrame(design_history)
    df_history['objective'] = objective_history
    df_history.to_csv("noria_optimization_history.csv", index=False)
    logging.info("Saved optimization history to noria_optimization_history.csv")

    # Plot objective history
    plt.figure(figsize=(10, 6))
    plt.plot(objective_history)
    plt.xlabel("Optimization Epoch")
    plt.ylabel("Objective Function Value (Negative Final Water Height)")
    plt.title("Noria Design Optimization Objective History")
    plt.grid(True)
    plt.savefig("noria_objective_history.png")
    plt.close()
    logging.info("Saved objective history plot to noria_objective_history.png")

    logging.info(f"Optimized Design Parameters: {current_design_params}")
