# GEMINI.md - Instructions for AI Agents on the SINN Project

## 1. Project Overview & Core Philosophy

This project is dedicated to building a **Systems-Informed Neural Network (SINN)** framework. This is a specialized, hierarchical type of Physics-Informed Neural Network (PINN) designed to model complex systems based on the principles of H.T. Odum's General Systems Theory.

**Core Philosophy:**
- **Modularity:** The framework must maintain a strict separation between pure physics definitions (`systems_library.py`), the generic ML engine (`pinn_framework.py`), and user-facing application scripts (`train_and_validate.py`, `inference.py`).
- **Transparency:** Physics models should be implemented in pure Python/NumPy where possible to be easily understood and verified (e.g., with an Euler solver) before being used to train a SINN.
- **Hierarchy & Scale:** The ultimate goal is to model systems hierarchically, allowing complex models to be composed of simpler sub-models, reflecting Odum's "Modeling for All Scales" concept.

## 2. Key Technologies

- **JAX:** For high-performance numerical computing and automatic differentiation (`grad`, `vmap`, `jit`).
- **Flax:** For creating neural network architectures as `nn.Module`s.
- **Optax:** For implementing optimizers and learning rate schedules.
- **NumPy:** For standard numerical operations and data manipulation.
- **Matplotlib:** For generating all plots and visualizations.
- **Pandas:** For creating and saving tabulated data (`.csv`).
- **Pytest:** For the testing framework.

## 3. Development Workflow & Testing

### 🚨 NON-NEGOTIABLE PRE-COMMIT REQUIREMENTS 🚨

The following commands **MUST** be run and **MUST** pass before any `git push`. This ensures code quality, correctness, and prevents regressions.

1.  **Install/Update Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Test Suite:**
    ```bash
    pytest
    ```

**ALL tests must pass with zero errors.**

### Testing Strategy

Our testing is divided into three categories, located in a `tests/` directory:

1.  **`tests/test_systems.py` (Unit Tests):**
    -   **Purpose:** To verify the correctness of the physics models in `systems_library.py` *in isolation*.
    -   **Example Test:** `test_euler_approximates_analytical()`: Asserts that the custom Euler solver's output is close to the known analytical solution for a given system.

2.  **`tests/test_training.py` (Integration Tests):**
    -   **Purpose:** To verify that the full training pipeline can successfully learn a solution that matches the ground truth.
    -   **Example Test:** `test_sinn_accuracy_against_analytical()`: Trains a SINN for a small number of epochs and asserts that its final prediction's Mean Squared Error (MSE) against the analytical solution is below a certain threshold.

3.  **`tests/test_regression.py` (Regression Tests):**
    -   **Purpose:** To ensure that code changes do not unexpectedly alter the final output of a trained model.
    -   **Workflow:**
        1.  When a model's output is deemed "correct," its output CSV is saved as a baseline (e.g., `tests/baseline_data/tank_inference.csv`).
        2.  The test `test_inference_csv_against_baseline()` will run the model and compare its new output CSV against the saved baseline.
        3.  The test fails if the outputs differ by more than a small tolerance, indicating a regression.

### Git Workflow

The following git workflow is **MANDATORY** for all development activities. Adhering to this ensures a clean history, proper tracking of progress, and seamless collaboration. Always check the GitHub Project TODO list for the next TODO item.

1.  **Start of Work Cycle (`GEMINI-CLI` behavior):**
    -   Before starting any new task or addressing an issue, `GEMINI-CLI` **MUST** execute:
        ```bash
        git fetch origin
        git pull origin main
        ```
        This ensures your local `main` branch is fully synchronized with the remote, preventing merge conflicts and working on stale code.

2.  **Branching for Each TODO/Issue:**
    -   Every new TODO item or GitHub Issue **MUST** correspond to a dedicated feature or bugfix branch.
    -   **Naming Convention:** Branches should be descriptive and directly link to the issue they address.
        -   For new features: `feature/descriptive-feature-name` (e.g., `feature/hierarchical-models`)
        -   For bug fixes: `bugfix/issue-number-short-description` (e.g., `bugfix/123-noria-torque-equation-fix`)
        -   Ensure that the branch name reflects the GitHub Issue title or a clear, concise description of the TODO item.
    -   **`GEMINI-CLI` Action:** When picking up a new issue, `GEMINI-CLI` will create and switch to a new branch. For example, if addressing GitHub Issue #456 "Implement new loss function":
        ```bash
        git checkout -b feature/456-new-loss-function
        ```

3.  **GitHub Project Management Integration:**
    -   When `GEMINI-CLI` picks up a GitHub Issue to work on (and creates the corresponding branch), it **MUST** update the status of that issue in the GitHub Project Management board to `In Progress`. This provides real-time visibility into active development.

4.  **Committing:**
    -   Use conventional commit messages (e.g., `feat(physics): Implement Noria torque model`, `fix(training): Resolve NaN issue in loss calculation`). This helps in generating changelogs and understanding the purpose of each commit.

5.  **Pull Requests (PRs):**
    -   All code **MUST** be merged into the `main` branch via a Pull Request.
    -   PRs **MUST** pass all automated checks (linting, testing) defined in the pre-commit requirements.
    -   A PR description should clearly state what the PR addresses, reference the corresponding GitHub Issue (e.g., "Closes #456"), and explain any significant design choices.

6.  **PR Approval and Merging:**
    -   After a Pull Request is created, `GEMINI-CLI` **MUST** pause its current task.
    -   Work on the next sequential task **MUST NOT** begin until the following conditions are met:
        1.  All automated GitHub Actions checks (e.g., `pytest`) have passed.
        2.  A human user has explicitly approved the Pull Request.
    -   Once the PR is approved and merged into the `main` branch, `GEMINI-CLI` may begin the next task, starting again from Step 1 (fetching and pulling the updated `main` branch).

## 4. Instructions for Implementing New Features

### A. Adding a New System Model (e.g., "Atwood's Machine")

1.  **Define the Physics (`systems_library.py`):**
    -   Create a new class, `AtwoodSystem`.
    -   In `__init__`, define its physical parameters (e.g., `m1`, `m2`, `g`).
    -   Implement the `get_derivative` method. This will be a system of two ODEs: one for velocity `dv/dt` and one for position `dy/dt`.
    -   Implement a `solve_euler` method to provide a ground-truth simulation.

2.  **Define the PINN Components (`train_atwood.py` or similar):**
    -   Create a new Flax Module `AtwoodPINN` that takes `t` as input and has **two outputs**: `v` and `y`.
    -   Create a new `atwood_loss_fn` that calculates the residuals for **both** `dv/dt` and `dy/dt` and sums their losses.

3.  **Create the Training Script (`train_atwood.py`):**
    -   Follow the structure of `train_and_validate.py`.
    -   Instantiate `AtwoodSystem` and `AtwoodPINN`.
    -   Use the `PINN_Framework` to train the model.
    -   Save the snapshot and output plots/tables comparing the SINN prediction to the Euler simulation.

4.  **Add Tests (`tests/`):**
    -   Add a `test_atwood_physics.py` to verify the Euler solver.
    -   Add a `test_atwood_training.py` to ensure the SINN can learn the solution with low error.

### B. Implementing Hierarchical Models (Future Goal)

-   **Objective:** Refactor `SystemModel` to be a recursive data structure.
-   **Key Task:** The `build_loss_function` factory must be able to "walk" the tree of a nested `SystemModel` and aggregate the loss residuals from all sub-components.
-   **PINN Architecture:** The `build_pinn_model` factory must be able to create a network with outputs corresponding to all state variables in the entire hierarchy.

### C. Implementing Data Assimilation (Future Goal)

-   **Objective:** Allow the training process to be guided by sparse experimental data.
-   **Key Task:** The `PINN_Framework.train` method and the loss function need to be modified to handle an additional `data` input. The loss will be a weighted sum of `Loss_Physics` and `Loss_Data`. The data generator must be updated to yield mixed batches.

## 5. Self-Correction and Improvement

As an AI agent, you are empowered to improve this project. If you identify a more robust, efficient, or clear way to implement any part of this framework, or if these instructions are ambiguous, you **MUST** propose an update to this `GEMINI.md` file. Your proposal should include the reason for the change, the specific changes, and the expected benefit.

## 6. Utilizing Inspiration Code (For `GEMINI-CLI`)

The `./inspiration` folder serves as a repository for Proof-of-Concept (PoC) code. This code may have been previously functional but does not adhere to the current repository standards (e.g., lack of proper modularity, testing, or adherence to the project's core philosophy).

**`GEMINI-CLI` Capability:**
When `GEMINI-CLI` is assigned a task that involves re-implementing or standardizing existing functionality, it **SHOULD** check the `./inspiration` folder for relevant PoC examples.

**Workflow for `GEMINI-CLI`:**

1.  **Identify Relevant PoC:** If a new task or GitHub Issue (e.g., "Standardize Tank Model Implementation") has a known PoC in `./inspiration` (e.g., `./inspiration/old_tank_model.py`), `GEMINI-CLI` should identify this file.
2.  **Analyze and Extract Logic:** `GEMINI-CLI` should analyze the PoC code to understand its core logic, algorithms, and mathematical formulations.
3.  **Rewrite to Standards:** The identified logic should then be re-written and integrated into the project following **ALL** the standards outlined in this `GEMINI.md` file, including:
    -   Adherence to Modularity principles (e.g., separating physics, ML engine, and application scripts).
    -   Creation of appropriate unit, integration, and regression tests.
    -   Following the Git Workflow (branching, committing, PRs).
    -   Ensuring code clarity, documentation, and efficiency.
4.  **No Direct Copy-Pasting:** Direct copy-pasting from `./inspiration` into the main codebase is **STRICTLY PROHIBITED**. The purpose of the `./inspiration` folder is to provide conceptual guidance, not deployable code. All code integrated into the main repository must be written from scratch or heavily refactored to meet current standards.
5.  **Remove/Archive PoC (Optional but Recommended):** Once the functionality from a PoC has been successfully re-implemented and integrated according to standards, the original PoC file in `./inspiration` can be considered for archival or removal if it no longer serves a purpose.