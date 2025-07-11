# train_noria.py

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import pandas as pd
import argparse

# --- Import our modular components ---
from src.systems_library import NoriaSystem
from src.pinn_framework import PINN_Framework
from src.pinn_builder import build_pinn_model # Import the new builder

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# SECTION 1: PROBLEM-SPECIFIC DEFINITIONS (The "JAX-ify" part)
# ==============================================================================

# --- The PINN Architecture for this problem (now dynamically built) ---

# --- The Physics-Informed Loss Function for this problem ---
def noria_loss_fn(params, model, t_coll, h_initial, omega_initial, system: NoriaSystem):
    """
    Calculates loss based on the Noria ODEs.
    """
    # Define functions for h(t) and omega(t) that use the current model parameters
    def state_fn(t_var):
        predictions = model.apply({'params': params}, t_var)
        return predictions

    # Get the derivative functions dh/dt and d_omega/dt using jax.grad and jax.vmap
    dh_dt_fn = jax.grad(lambda t_var: model.apply({'params': params}, t_var)[0])
    d_omega_dt_fn = jax.grad(lambda t_var: model.apply({'params': params}, t_var)[1])

    # --- Physics Loss ---
    # Evaluate the model and its derivatives at all collocation points
    predictions_coll = jax.vmap(model.apply, in_axes=(None, 0))({'params': params}, t_coll)
    h_pred_coll = predictions_coll[:, 0]
    omega_pred_coll = predictions_coll[:, 1]
    dh_dt_pred_coll = jax.vmap(dh_dt_fn)(t_coll)
    d_omega_dt_pred_coll = jax.vmap(d_omega_dt_fn)(t_coll)
    
    # Get the "right-hand side" of the ODEs from our single source of truth
    # The system.get_derivative expects a state array [h, omega]
    physics_rhs = jax.vmap(system.get_derivative)(jnp.stack([h_pred_coll, omega_pred_coll], axis=-1))
    
    # The residuals are the differences between the two sides of the ODEs
    residual_h = dh_dt_pred_coll - physics_rhs[:, 0]
    residual_omega = d_omega_dt_pred_coll - physics_rhs[:, 1]

    loss_physics = jnp.mean(residual_h**2) + jnp.mean(residual_omega**2)

    # --- Initial Condition Loss ---
    initial_predictions = model.apply({'params': params}, jnp.array([0.0]))
    h_pred_initial = initial_predictions[0]
    omega_pred_initial = initial_predictions[1]
    
    # Return the weighted sum of the losses
    return loss_physics + 100.0 * (loss_initial_h + loss_initial_omega)

# --- The Data Generator for this problem ---
def noria_data_generator(h0, omega0, collocation_size=200, t_max=50.0):
    """Yields batches of random collocation points and initial conditions."""
    key = jax.random.PRNGKey(0)
    while True:
        key, subkey = jax.random.split(key)
        t_coll = jax.random.uniform(subkey, (collocation_size,)) * t_max
        # Yield a tuple of the DYNAMIC arguments for the loss function
        yield (t_coll, h0, omega0)

# ==============================================================================
# SECTION 2: MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a PINN for the Noria System.')
    parser.add_argument('--epochs', type=int, default=75000, help='Number of training epochs.')
    parser.add_argument('--inference-only', action='store_true', help='Run inference using a pre-trained model snapshot.')
    parser.add_argument('--snapshot-dir', type=str, default='./snapshots/noria_model_snapshot', help='Directory to save or load the model snapshot.')
    args = parser.parse_args()

    # 1. DEFINE THE SPECIFIC SYSTEM TO SOLVE
    # Instantiate the pure physics model from our library
    system_instance = NoriaSystem(Q_in=1.0, k_q=0.1, k_tau=0.5, k_friction=0.05, I=10.0, h0=1.0, omega0=0.0)
    
    # 2. INSTANTIATE THE PINN ARCHITECTURE
    pinn_model_arch = build_pinn_model(output_dim=2) # Noria has 2 outputs (h, omega)
    dummy_t = jnp.ones((1,))
    
    # --- Training or Inference Logic ---
    if args.inference_only:
        logging.info(f"Loading model from snapshot: {args.snapshot_dir}")
        pinn_solver = PINN_Framework.load_snapshot(model=pinn_model_arch, dummy_inputs=(dummy_t,), checkpoint_dir=args.snapshot_dir)
        loss_history = [] # No training history in inference-only mode
    else:
        # 3. INSTANTIATE THE GENERIC FRAMEWORK
        pinn_solver = PINN_Framework(
            model=pinn_model_arch,
            loss_fn=noria_loss_fn,
            dummy_inputs=(dummy_t,),
            static_loss_args={'system': system_instance}
        )

        # 4. TRAIN THE MODEL
        data_gen = noria_data_generator(h0=system_instance.h0, omega0=system_instance.omega0)
        
        loss_history = pinn_solver.train(
            training_data_generator=data_gen,
            num_epochs=args.epochs
        )
        
        # 5. SAVE THE TRAINED MODEL SNAPSHOT
        pinn_solver.save_snapshot(directory=args.snapshot_dir, step=args.epochs)
    
    # 6. VALIDATE AND VISUALIZE TRAINING RESULTS
    # Get ground truth from our verified systems library
    t_plot = np.linspace(0, 50, 501)
    # NoriaSystem.solve_euler returns (time_steps, state_values) where state_values is (time_steps, 2)
    _, analytical_states = system_instance.solve_euler(t_max=50.0, dt=0.1)
    h_analytical = analytical_states[:, 0]
    omega_analytical = analytical_states[:, 1]
    
    # Get prediction from the trained PINN
    predictor = pinn_solver.get_predictor()
    predictions_pinn = jax.vmap(predictor)(t_plot)
    h_pinn = predictions_pinn[:, 0]
    omega_pinn = predictions_pinn[:, 1]

    # Plot the final comparison for h
    plt.figure(figsize=(12, 8))
    plt.plot(t_plot, h_analytical, 'k-', linewidth=3, label='Ground Truth h (Euler)')
    plt.plot(t_plot, h_pinn, 'r--', linewidth=2, label='PINN Prediction h')
    plt.scatter([0.0], [system_instance.h0], color='blue', zorder=5, s=100, label='Initial Condition h')
    plt.xlabel('Time (t)')
    plt.ylabel('Water Height (h)')
    plt.title("Noria System: PINN vs. Euler for Water Height (h)")
    plt.legend()
    plt.grid(True)
    plt.savefig("noria_h_comparison.png")
    plt.close()
    logging.info("Saved water height comparison plot to noria_h_comparison.png")

    # Plot the final comparison for omega
    plt.figure(figsize=(12, 8))
    plt.plot(t_plot, omega_analytical, 'k-', linewidth=3, label='Ground Truth omega (Euler)')
    plt.plot(t_plot, omega_pinn, 'r--', linewidth=2, label='PINN Prediction omega')
    plt.scatter([0.0], [system_instance.omega0], color='blue', zorder=5, s=100, label='Initial Condition omega')
    plt.xlabel('Time (t)')
    plt.ylabel('Angular Velocity (omega)')
    plt.title("Noria System: PINN vs. Euler for Angular Velocity (omega)")
    plt.legend()
    plt.grid(True)
    plt.savefig("noria_omega_comparison.png")
    plt.close()
    logging.info("Saved angular velocity comparison plot to noria_omega_comparison.png")

    # Save training history to CSV
    pd.DataFrame({'loss': loss_history}).to_csv("noria_training_history.csv", index=False)
    logging.info("Saved training history to noria_training_history.csv")

    # 7. Save inference data to CSV
    inference_data = pd.DataFrame({
        't': t_plot,
        'h_analytical': h_analytical,
        'h_pinn': h_pinn,
        'omega_analytical': omega_analytical,
        'omega_pinn': omega_pinn
    })
    inference_data.to_csv("noria_inference_data.csv", index=False)
    logging.info("Saved inference data to noria_inference_data.csv")