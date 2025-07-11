# tests/test_training_and_inference.py

import pytest
import subprocess
import os
import numpy as np
import pandas as pd

import tempfile

# Define the path to the main application script
MAIN_APP_PATH = "src/main.py"

@pytest.fixture(scope="module")
def run_main_app():
    """
    Runs the main application script as a subprocess and cleans up generated files.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        snapshot_dir = os.path.join(temp_dir, "snapshots")
        # Ensure the snapshots directory exists for the app to save to
        os.makedirs(snapshot_dir, exist_ok=True)
        # Run the main application script
        try:
            # Execute the script. Capture output for debugging if needed.
            result = subprocess.run(
                ["python", MAIN_APP_PATH, "--epochs", "100", "--snapshot-dir", snapshot_dir],
                capture_output=True, text=True, check=True
            )
            print(result.stdout)
            print(result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error running main.py: {e}")
            print(e.stdout)
            print(e.stderr)
            pytest.fail(f"main.py failed with exit code {e.returncode}")

        # Yield control to the tests, passing the snapshot_dir
        yield snapshot_dir

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

def test_main_app_generates_expected_files(run_main_app):
    """
    Tests that the main application script generates all expected output files.
    """
    snapshot_dir = run_main_app # The fixture now yields the snapshot_dir
    expected_files = [
        "tank_training_comparison.png",
        "training_history.csv",
        "tank_parametric_inference.png",
        "inference_data.csv",
    ]
    for f in expected_files:
        assert os.path.exists(f), f"Expected file {f} was not generated."
    
    # Check for snapshot directory
    assert os.path.exists(os.path.join(snapshot_dir, "checkpoint_100")), "Snapshot directory was not created."

def test_training_loss_decreases(run_main_app):
    """
    Tests that the training loss recorded in training_history.csv decreases.
    """
    _ = run_main_app # Consume the fixture yield
    df = pd.read_csv("training_history.csv")
    losses = df['loss'].values
    
    # Only check for loss decrease if there are enough epochs to show a trend
    if len(losses) > 1:
        assert losses[-1] < losses[0] * 0.1, "Training loss did not decrease sufficiently."

def test_inference_data_format(run_main_app):
    """
    Tests the format and basic content of the inference_data.csv file.
    """
    _ = run_main_app # Consume the fixture yield
    df = pd.read_csv("inference_data.csv")
    assert not df.empty, "Inference data CSV is empty."
    assert all(col in df.columns for col in ['J', 't', 'Q_analytical', 'Q_pinn', 'error'])
    assert df['J'].nunique() > 1, "Inference data should contain multiple J values."
