# tests/test_training_and_inference.py

import pytest
import subprocess
import os
import numpy as np
import pandas as pd

# Define the path to the main application script
MAIN_APP_PATH = "src/main.py"

@pytest.fixture(scope="module")
def run_main_app():
    """
    Runs the main application script as a subprocess and cleans up generated files.
    """
    # Ensure the snapshots directory exists for the app to save to
    os.makedirs("snapshots", exist_ok=True)

    # Run the main application script
    # We use a smaller number of epochs for testing to speed it up
    # This requires modifying main.py to accept an epoch argument or similar
    # For now, we'll assume it runs quickly enough or we'll mock it.
    # For a real test, you'd pass a flag or environment variable to main.py
    # to run for fewer epochs.
    # For this test, we'll just run it as is and hope it's fast enough.
    try:
        # Execute the script. Capture output for debugging if needed.
        result = subprocess.run(
            ["python", MAIN_APP_PATH],
            capture_output=True, text=True, check=True
        )
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running main.py: {e}")
        print(e.stdout)
        print(e.stderr)
        pytest.fail(f"main.py failed with exit code {e.returncode}")

    # Yield control to the tests
    yield

    # Teardown: Clean up generated files and directories
    generated_files = [
        "tank_training_comparison.png",
        "training_history.csv",
        "tank_parametric_inference.png",
        "inference_data.csv",
    ]
    for f in generated_files:
        if os.path.exists(f):
            os.remove(f)
    
    # Clean up snapshot directory
    snapshot_dir = "snapshots/tank_model_snapshot"
    if os.path.exists(snapshot_dir):
        import shutil
        shutil.rmtree(snapshot_dir)

def test_main_app_generates_expected_files(run_main_app):
    """
    Tests that the main application script generates all expected output files.
    """
    expected_files = [
        "tank_training_comparison.png",
        "training_history.csv",
        "tank_parametric_inference.png",
        "inference_data.csv",
    ]
    for f in expected_files:
        assert os.path.exists(f), f"Expected file {f} was not generated."
    
    # Check for snapshot directory
    assert os.path.exists("snapshots/tank_model_snapshot"), "Snapshot directory was not created."

def test_training_loss_decreases(run_main_app):
    """
    Tests that the training loss recorded in training_history.csv decreases.
    """
    df = pd.read_csv("training_history.csv")
    losses = df['loss'].values
    
    # Assert that the last loss is significantly lower than the first loss
    # This is a basic check for training progress
    assert losses[-1] < losses[0] * 0.1, "Training loss did not decrease sufficiently."

def test_inference_data_format(run_main_app):
    """
    Tests the format and basic content of the inference_data.csv file.
    """
    df = pd.read_csv("inference_data.csv")
    assert not df.empty, "Inference data CSV is empty."
    assert all(col in df.columns for col in ['J', 't', 'Q_analytical', 'Q_pinn', 'error'])
    assert df['J'].nunique() > 1, "Inference data should contain multiple J values."
