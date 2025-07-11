# SINN Framework - Development TODO List

## Phase 1: Core Framework (Complete)
- [x] Create `systems_library.py` for pure physics definitions.
- [x] Implement `TankSystem` with analytical and Euler solvers.
- [x] Create generic `pinn_framework.py`.
- [x] Create `main_app.py` to train and validate the TANK model.
- [x] Create `generate_report.py` to produce a PDF summary.

## Phase 2: Hierarchical Modeling
- [ ] Refactor `systems_library.py` to allow `SystemModel` to contain other `SystemModel`s.
- [ ] Create a `pinn_builder.py` module.
- [ ] Implement `build_pinn_model` to dynamically create architectures from a hierarchical system.
- [ ] Implement `build_loss_function` to recursively construct the loss from a hierarchical system.

## Phase 3: Data Assimilation
- [ ] Update `PINN_Framework` to accept an optional `data` argument.
- [ ] Update the loss function builder to include a weighted `Loss_Data` term.
- [ ] Update the data generator to produce mixed batches of collocation and real data points.

## Phase 4: Noria Application
- [ ] Implement the `NoriaSystem` class in `systems_library.py` with all relevant physics (torque, backwater, etc.).
- [ ] Create a `train_noria.py` application script.
- [ ] Create an `optimize_design.py` script that loads the trained Noria SINN and uses `jax.grad` to find the optimal `arc_length`.