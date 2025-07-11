# SINN: A Systems-Informed Neural Network Framework

## 1. Project Overview

This project implements a **Systems-Informed Neural Network (SINN)**, a modular and hierarchical framework for building digital twins of complex physical and biological systems. It combines the principles of **General Systems Theory**, as pioneered by H.T. Odum, with the power of modern, physics-informed machine learning.

The core idea is to train neural networks not just on low-level Partial Differential Equations (PDEs), but on the **lumped-parameter Ordinary Differential Equations (ODEs)** that describe the emergent, system-level behavior of storages and flows.

This repository contains:
- A generic, reusable framework for training SINNs (`pinn_framework.py`).
- A library for defining the pure physics of systems (`systems_library.py`).
- A demonstration of how to use the framework to solve Odum's classic TANK model (`main_app.py`).

## 2. How to Run

1.  **Install Dependencies:**
    ```bash
    pip install jax flax optax matplotlib numpy pandas scikit-learn
    ```

2.  **Train the Model:**
    Navigate to the `src/` directory and run the main application. This will train the model and generate output plots and data.
    ```bash
    cd src
    python main_app.py
    ```

3.  **Generate the Report (Optional):**
    To generate a full PDF report of the results, you will need a LaTeX distribution (e.g., TeX Live).
    ```bash
    # Make sure you are still in the src/ directory
    python generate_report.py
    pdflatex SINN_Report.tex
    pdflatex SINN_Report.tex
    ```

## 3. Project Goals

The ultimate goal is to extend this framework to model complex, multi-scale systems, such as a Noria water wheel interacting with a hydraulic flume, and to use the resulting digital twin for automated design optimization. For more details, see `DESIGN.md` and `REQUIREMENTS.md`.# unsw-phd-simm
