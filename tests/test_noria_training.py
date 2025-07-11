import pytest
import subprocess
import os
import pandas as pd
import tempfile
import shutil

# Define the path to the Noria training script
NORIA_TRAINING_SCRIPT_PATH = os.path.abspath("train_noria.py")

@pytest.fixture(scope="module")
def run_noria_training_app():
    """
    Runs the Noria training script as a subprocess and cleans up generated files.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        snapshot_dir = os.path.join(temp_dir, "snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        
        try:
            # Execute the script with a small number of epochs for testing
            result = subprocess.run(
                ["python", NORIA_TRAINING_SCRIPT_PATH, "--epochs", "100", "--snapshot-dir", snapshot_dir],
                capture_output=True, text=True, check=True
            )
            print(result.stdout)
            print(result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error running {NORIA_TRAINING_SCRIPT_PATH}: {e}")
            print(e.stdout)
            print(e.stderr)
            pytest.fail(f"{NORIA_TRAINING_SCRIPT_PATH} failed with exit code {e.returncode}")

        yield snapshot_dir

    # Teardown: Clean up generated files
    generated_files = [
        "noria_h_comparison.png",
        "noria_omega_comparison.png",
        "noria_training_history.csv",
        "noria_inference_data.csv",
    ]
    for f in generated_files:
        if os.path.exists(f):
            os.remove(f)

def test_noria_app_generates_expected_files(run_noria_training_app):
    """
    Tests that the Noria training script generates all expected output files.
    """
    snapshot_dir = run_noria_training_app
    expected_files = [
        "noria_h_comparison.png",
        "noria_omega_comparison.png",
        "noria_training_history.csv",
        "noria_inference_data.csv",
    ]
    for f in expected_files:
        assert os.path.exists(f), f"Expected file {f} was not generated."
    
    # Check for snapshot directory
    assert os.path.exists(os.path.join(snapshot_dir, "checkpoint_100")), "Noria snapshot directory was not created."

def test_noria_training_loss_decreases(run_noria_training_app):
    """
    Tests that the training loss recorded in noria_training_history.csv decreases.
    """
    _ = run_noria_training_app
    df = pd.read_csv("noria_training_history.csv")
    losses = df['loss'].values
    
    if len(losses) > 1:
        assert losses[-1] < losses[0] * 0.1, "Noria training loss did not decrease sufficiently."

def test_noria_inference_data_format(run_noria_training_app):
    """
    Tests the format and basic content of the noria_inference_data.csv file.
    """
    _ = run_noria_training_app
    df = pd.read_csv("noria_inference_data.csv")
    assert not df.empty, "Noria inference data CSV is empty."
    assert all(col in df.columns for col in ['t', 'h_analytical', 'h_pinn', 'omega_analytical', 'omega_pinn'])
    assert df['t'].nunique() > 1, "Noria inference data should contain multiple t values."
