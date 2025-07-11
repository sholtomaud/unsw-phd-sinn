# SINN Framework: Design Philosophy

## 1. Core Principle: From PINN to SINN

While a standard **Physics-Informed Neural Network (PINN)** can be informed by any differential equation, our **Systems-Informed Neural Network (SINN)** is specifically designed to leverage the principles of **General Systems Theory**.

Our philosophy is that many complex phenomena are best understood through the emergent, top-down dynamics of their aggregate components (storages and flows), as described by Ordinary Differential Equations (ODEs). This contrasts with a purely bottom-up approach that relies on solving complex Partial Differential Equations (PDEs) for every point in space.

This framework is designed to bridge these two worlds.

## 2. Architectural Goals

Our design is guided by three main intentions:

### A. Modularity and Separation of Concerns
The project is split into three distinct logical parts to ensure clarity, testability, and reusability:
- **`systems_library.py`**: The **Physics Definition**. This module contains pure Python/NumPy classes that define the governing equations of a system. It is completely independent of any machine learning code and serves as the "single source of truth" for the physics.
- **`pinn_framework.py`**: The **Generic Engine**. This module contains the `PINN_Framework` class, a problem-agnostic tool that knows how to train, save, and load neural networks based on a given loss function.
- **`main_app.py`**: The **Application Layer**. This is the user's script, where a specific system from the library is chosen and paired with a PINN architecture. It uses the framework to orchestrate the training and validation process.

### B. Hierarchical and Multi-Scale Modeling
Real-world systems are hierarchical. A key future goal of this framework is to support this reality. The architecture is designed to eventually allow `SystemModel` objects to be composed of other `SystemModel`s. This will enable the creation of multi-scale digital twins where the loss function simultaneously enforces physical laws at different levels of abstraction (e.g., both the system-level ODE and the component-level PDE).

### C. From Simulation to Optimization
The final intention is to move beyond simple simulation and create a tool for **automated design**. By ensuring our SINN models are fully differentiable, we can use gradient-based methods to optimize physical design parameters (e.g., the bucket size of a Noria wheel) to achieve a specific goal (e.g., maximum power output). This eliminates the need for slow, external, "black-box" optimization loops (like `pymoo`).