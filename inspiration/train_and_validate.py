# ==============================================================================
# main.py
#
# Main application script to:
# 1. Train the parametric SINN for the TANK system.
# 2. Save the trained model snapshot.
# 3. Validate the model against analytical and Euler solutions.
# 4. Save all diagnostic plots and data tables for the report.
# ==============================================================================

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.struct import field
import matplotlib.pyplot as plt
import numpy as np
import logging
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from functools import partial

# --- Import our modular components ---
from pinn_framework import PINN_Framework
from systems_library import TankSystem

# --- 1. PROBLEM-SPECIFIC DEFINITIONS ---
class ParametricTankPINN(nn.Module):
    """A PINN that learns the function Q(t, J, k)."""
    features: list[int] = field(default_factory=lambda: [32, 32, 32, 1])
    @nn.compact
    def __call__(self, t, J, k):
        t_vec, J_vec, k_vec = jnp.asarray(t).reshape(-1, 1), jnp.asarray(J).reshape(-1, 1), jnp.asarray(k).reshape(-1, 1)
        inputs = jnp.concatenate([t_vec, J_vec, k_vec], axis=1)
        x = inputs
        for i, feat in enumerate(self.features):
            x = nn.Dense(features=feat)(x)
            if i < len(self.features) - 1: x = nn.tanh(x)
        return nn.softplus(x).squeeze()

def tank_loss_fn(params, model, t_coll, J_coll, k_coll, t_initial, Q_initial):
    """Physics-informed loss for the parametric TANK system."""
    Q_fn = lambda t, J, k: model.apply({'params': params}, t, J, k)
    dQ_dt_fn = jax.grad(Q_fn, argnums=0)
    Q_pred_coll = jax.vmap(Q_fn)(t_coll, J_coll, k_coll)
    dQ_dt_pred_coll = jax.vmap(dQ_dt_fn)(t_coll, J_coll, k_coll)
    physics_rhs = jax.vmap(system_instance.get_derivative, in_axes=(0, None))(Q_pred_coll, None)
    residual_physics = dQ_dt_pred_coll - physics_rhs
    loss_physics = jnp.mean(residual_physics**2)
    Q_pred_initial = jax.vmap(Q_fn)(t_initial, J_coll, k_coll)
    loss_initial = jnp.mean((Q_pred_initial - Q_initial)**2)
    return loss_physics + 100.0 * loss_initial

def tank_data_generator(Q0, collocation_size=200, t_max=50.0):
    """Yields batches of random collocation points over ranges of t, J, and k."""
    key = jax.random.PRNGKey(0)
    while True:
        key, kt, kJ, kk = jax.random.split(key, 4)
        t_coll = jax.random.uniform(kt, (collocation_size,)) * t_max
        J_coll = jax.random.uniform(kJ, (collocation_size,), minval=1.0, maxval=10.0)
        k_coll = jax.random.uniform(kk, (collocation_size,), minval=0.05, maxval=0.2)
        t_initial = jnp.zeros_like(J_coll)
        Q_initial = jnp.full_like(J_coll, Q0)
        yield (t_coll, J_coll, k_coll, t_initial, Q_initial)

# --- 2. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # --- Part 1: Training ---
    logging.info("--- Starting SINN Training ---")
    system_instance = TankSystem(J=2.0, k=0.1, Q0=1.0)
    pinn_model_arch = ParametricTankPINN()
    dummy_t, dummy_J, dummy_k = jnp.ones((1,)), jnp.ones((1,)), jnp.ones((1,))
    
    pinn_solver = PINN_Framework(
        model=pinn_model_arch,
        loss_fn=tank_loss_fn,
        dummy_inputs=(dummy_t, dummy_J, dummy_k)
    )
    
    num_epochs = 40000
    data_gen = tank_data_generator(Q0=system_instance.Q0)
    
    # Fixed: Remove static_loss_args parameter
    loss_history = pinn_solver.train(data_gen, num_epochs=num_epochs)
    
    pinn_solver.save_snapshot(directory='./tank_model_snapshot', step=num_epochs)
    logging.info("--- Training Complete ---")

    # --- Part 2: Validation and Output Generation ---
    logging.info("--- Generating Final Diagnostics and Outputs ---")
    
    # A) Calculate Final Metrics
    t_val = np.linspace(0, 50, 500)
    Q_analytical_val = system_instance.solve_analytical(t_val)
    predictor = pinn_solver.get_predictor()
    J_val = jnp.full_like(t_val, system_instance.J)
    k_val = jnp.full_like(t_val, system_instance.k)
    Q_pinn_val = jax.vmap(predictor)(t_val, J_val, k_val)
    final_mae = mean_absolute_error(Q_analytical_val, Q_pinn_val)
    final_r2 = r2_score(Q_analytical_val, Q_pinn_val)
    
    logging.info(f"Final MAE: {final_mae:.6f}")
    logging.info(f"Final R²: {final_r2:.6f}")
    
    # B) Save Training History and Metrics to CSV
    history_df = pd.DataFrame({
        'epoch': np.arange(1, len(loss_history) + 1) * 1000,
        'loss': loss_history,
        'val_mae': [np.nan] * (len(loss_history) - 1) + [final_mae],
        'val_r2': [np.nan] * (len(loss_history) - 1) + [final_r2]
    })
    history_df.to_csv('training_history.csv', index=False, float_format='%.6f')
    logging.info("Saved training history to training_history.csv")

    # C) Save Loss Curve Plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(history_df['epoch'], history_df['loss'], 'k-')
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss (Log Scale)')
    ax1.set_title('Training Loss Curve')
    ax1.grid(True, which="both", ls="--")
    plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    logging.info("Saved training loss curve to loss_curve.png")

    # D) Save Comparison Plot
    t_euler, Q_euler = system_instance.solve_euler(t_max=50, dt=0.1)
    fig2, ax2 = plt.subplots(figsize=(14, 9))
    ax2.plot(t_val, Q_analytical_val, 'k-', linewidth=4, alpha=0.8, label='Analytical Solution (Ground Truth)')
    ax2.plot(t_val, Q_pinn_val, 'c--', linewidth=2.5, label='SINN Prediction')
    ax2.plot(t_euler, Q_euler, 'r:', linewidth=2, label='Odum-Euler Simulation')
    ax2.scatter([0.0], [system_instance.Q0], color='blue', zorder=10, s=150, edgecolors='white', label='Initial Condition')
    ax2.set_xlabel('Time (t)', fontsize=14)
    ax2.set_ylabel('Storage (Q)', fontsize=14)
    ax2.set_title(f"Model Comparison for J={system_instance.J}, k={system_instance.k}", fontsize=18)
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--')
    plt.savefig('final_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    logging.info("Saved final comparison plot to final_comparison.png")

    # E) Save Inference Data Table to CSV
    # Fixed: Convert to numpy array first
    t_points_for_table = np.array([0.0, 5.0, 10.0, 25.0, 50.0])
    analytical_Q_table = system_instance.solve_analytical(t_points_for_table)
    J_table = jnp.full_like(t_points_for_table, system_instance.J)
    k_table = jnp.full_like(t_points_for_table, system_instance.k)
    pinn_Q_table = predictor(t_points_for_table, J_table, k_table)
    
    # Find closest Euler solution points
    euler_indices = [np.argmin(np.abs(np.array(t_euler) - t)) for t in t_points_for_table]
    euler_Q_table = [Q_euler[i] for i in euler_indices]
    
    inference_df = pd.DataFrame({
        'Time (t)': t_points_for_table,
        'Analytical Q': analytical_Q_table,
        'Euler Q': euler_Q_table,
        'SINN Predicted Q': np.array(pinn_Q_table),
        'SINN Abs. Error': np.abs(analytical_Q_table - np.array(pinn_Q_table))
    })
    inference_df.to_csv('inference_data.csv', index=False, float_format='%.4f')
    logging.info("Saved inference data to inference_data.csv")
    
    # F) Print Summary Statistics
    logging.info("--- Training Summary ---")
    logging.info(f"Total epochs: {num_epochs}")
    logging.info(f"Final loss: {loss_history[-1]:.6f}")
    logging.info(f"Final MAE: {final_mae:.6f}")
    logging.info(f"Final R²: {final_r2:.6f}")
    logging.info("--- All outputs saved successfully ---")