# ==============================================================================
# pinn_framework.py (v5 - Final, Definitive, Corrected Version)
# ==============================================================================

import jax
import optax
from flax import linen as nn
from flax.training import checkpoints
import numpy as np
import logging
import os
from typing import Any, Callable, Tuple, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

LossFn = Callable[[Any, nn.Module, Any], float]

class PINN_Framework:
    def __init__(self, model: nn.Module, loss_fn: LossFn, dummy_inputs: Tuple, static_loss_args: Dict = {}, learning_rate: float = 1e-3):
        self.model = model
        self.loss_fn = loss_fn
        self.static_loss_args = static_loss_args # Store static args
        self.key = jax.random.PRNGKey(42)
        
        self.params = model.init(self.key, *dummy_inputs)['params']
        
        lr_schedule = optax.exponential_decay(init_value=learning_rate, transition_steps=5000, decay_rate=0.9)
        self.optimizer = optax.adam(learning_rate=lr_schedule)
        self.opt_state = self.optimizer.init(self.params)
        
        # --- THIS IS THE KEY CORRECTION ---
        # The JIT-compiled function is created ONCE during initialization.
        self.train_step = self._make_train_step()

    def _make_train_step(self) -> Callable:
        # Wrap the loss function to handle static arguments correctly
        def wrapped_loss_fn(params, model, dynamic_args):
            # The static args are now part of the closure, which is safe
            return self.loss_fn(params, model, *dynamic_args, **self.static_loss_args)

        @jax.jit
        def step_fn(params, opt_state, dynamic_args):
            loss_val, grads = jax.value_and_grad(wrapped_loss_fn, argnums=0)(params, self.model, dynamic_args)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_val
        
        return step_fn

    def train(self, training_data_generator, num_epochs: int):
        logging.info(f"Starting training for {num_epochs} epochs...")
        loss_history = []

        for epoch in range(num_epochs):
            batch_args = next(training_data_generator)
            self.params, self.opt_state, loss_val = self.train_step(self.params, self.opt_state, batch_args)
            
            if (epoch + 1) % 1000 == 0:
                logging.info(f"Epoch {epoch+1:5d} | Loss: {loss_val:.6f}")
                loss_history.append(np.array(loss_val))
        
        logging.info("Training finished.")
        return loss_history

    # ... (save_snapshot, load_snapshot, get_predictor are the same) ...
    def save_snapshot(self, directory: str, step: int):
        abs_path = os.path.abspath(directory)
        os.makedirs(abs_path, exist_ok=True)
        checkpoints.save_checkpoint(ckpt_dir=abs_path, target=self.params, step=step, overwrite=True)
        logging.info(f"Model snapshot saved to '{abs_path}'")

    @classmethod
    def load_snapshot(cls, model: nn.Module, dummy_inputs: Tuple, checkpoint_dir: str):
        temp_framework = cls(model, lambda: None, dummy_inputs)
        abs_path = os.path.abspath(checkpoint_dir)
        loaded_params = checkpoints.restore_checkpoint(ckpt_dir=abs_path, target=temp_framework.params)
        if loaded_params is None:
            raise FileNotFoundError(f"No checkpoint found in '{abs_path}'.")
        temp_framework.params = loaded_params
        logging.info(f"Model snapshot successfully loaded from '{abs_path}'")
        return temp_framework

    def get_predictor(self):
        @jax.jit
        def predictor(params, *args):
            return self.model.apply({'params': params}, *args)
        from functools import partial
        return partial(predictor, self.params)