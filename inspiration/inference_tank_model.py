# ==============================================================================
# inference_tank_model.py
#
# Loads a pre-trained SINN snapshot for the TANK model and uses it for
# parametric inference, comparing its predictions against the ground truth
# across a range of input parameters.
# ==============================================================================

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import logging

# --- Import our modular components ---
from pinn_framework import PINN_Framework
from systems_library import TankSystem
from train_tank_model import TankPINN # We need the architecture definition

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # 1. LOAD THE TRAINED PINN SNAPSHOT
    pinn_model_arch = TankPINN()
    dummy_t = jnp.ones((1,))
    
    try:
        pinn_solver = PINN_Framework.load_snapshot(
            model=pinn_model_arch,
            dummy_inputs=(dummy_t,),
            checkpoint_dir='../snapshots/tank_model_snapshot' # Relative path from src/
        )
    except FileNotFoundError as e:
        logging.error(f"ERROR: {e}\nPlease run 'python train_tank_model.py' first.")
        exit()

    # 2. GET THE PREDICTOR FUNCTION
    predictor = pinn_solver.get_predictor()

    # 3. RUN PARAMETRIC INFERENCE AND PLOT
    plt.figure(figsize=(14, 9))
    plt.title("Parametric Inference: SINN Performance Across Different Inflow Rates (J)", fontsize=18)
    
    J_values_to_test = [1.0, 2.0, 4.0, 6.0]
    k_fixed = 0.1
    Q0 = 1.0
    t_plot = np.linspace(0, 50, 500)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(J_values_to_test)))

    for i, J_test in enumerate(J_values_to_test):
        # Instantiate the ground truth system for this J
        system = TankSystem(J=J_test, k=k_fixed, Q0=Q0)
        Q_analytical = system.solve_analytical(t_plot)
        
        # Get SINN predictions
        # The original PINN was trained on a single J, so this tests generalization
        # A parametrically trained PINN would be even more accurate here.
        Q_pinn = jax.vmap(predictor)(t_plot) # Note: our simple PINN doesn't take J/k as input yet

        # Calculate the error
        error = np.abs(Q_analytical - Q_pinn)

        # Plot the analytical solution and the PINN prediction
        line, = plt.plot(t_plot, Q_analytical, '-', color=colors[i], lw=2, label=f'Analytical (J={J_test})')
        plt.plot(t_plot, Q_pinn, '--', color=colors[i], lw=2, label=f'SINN (J={J_test})')
        
        # --- NEW: Add error shading between the curves ---
        plt.fill_between(t_plot, Q_analytical, Q_pinn, color=colors[i], alpha=0.2, label=f'Error (J={J_test})')

    plt.xlabel('Time (t)', fontsize=14)
    plt.ylabel('Storage (Q)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--')
    plt.savefig("parametric_inference.png", dpi=300, bbox_inches='tight')
    plt.show()
    logging.info("Saved parametric inference plot to parametric_inference.png")