import pytest
import subprocess
import os
import pandas as pd
import tempfile
import shutil

# Define the path to the Noria optimization script
NORIA_OPTIMIZATION_SCRIPT_PATH = os.path.abspath("noria_optimization.py")

@pytest.fixture(scope="module")
def run_noria_optimization_app():
    """
    Runs the Noria optimization script as a subprocess and cleans up generated files.
    """
    # Ensure the snapshot directory for the Noria model exists
    # This assumes the Noria training script has been run at least once
    # to create the baseline snapshot.
    noria_snapshot_dir = "./snapshots/noria_model_snapshot"
    if not os.path.exists(noria_snapshot_dir):
        # If the snapshot doesn't exist, we need to create a dummy one or train it.
        # For testing, we'll create a minimal dummy structure.
        # In a real scenario, you'd ensure the training script runs first.
        os.makedirs(noria_snapshot_dir, exist_ok=True)
        # Create a dummy checkpoint file to avoid FileNotFoundError
        # This is a placeholder and won't contain actual model parameters.
        with open(os.path.join(noria_snapshot_dir, "checkpoint_75000"), "w") as f:
            f.write("{}")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Change to temp_dir to ensure generated files are in a clean location
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Execute the script with a small number of epochs for testing
            result = subprocess.run(
                ["python", NORIA_OPTIMIZATION_SCRIPT_PATH, "--epochs", "10"], # Small epochs for quick test
                capture_output=True, text=True, check=True
            )
            print(result.stdout)
            print(result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error running {NORIA_OPTIMIZATION_SCRIPT_PATH}: {e}")
            print(e.stdout)
            print(e.stderr)
            pytest.fail(f"{NORIA_OPTIMIZATION_SCRIPT_PATH} failed with exit code {e.returncode}")
        finally:
            os.chdir(original_cwd) # Change back to original directory

        yield temp_dir

    # Teardown: Clean up generated files (from original_cwd)
    generated_files = [
        "noria_optimization_history.csv",
        "noria_objective_history.png",
    ]
    for f in generated_files:
        if os.path.exists(f):
            os.remove(f)

    # Clean up dummy snapshot if created by this test
    if not os.path.exists(noria_snapshot_dir) or os.path.isdir(noria_snapshot_dir) and not os.listdir(noria_snapshot_dir):
        if os.path.exists(noria_snapshot_dir):
            shutil.rmtree(noria_snapshot_dir)

def test_noria_optimization_generates_expected_files(run_noria_optimization_app):
    """
    Tests that the Noria optimization script generates all expected output files.
    """
    temp_dir = run_noria_optimization_app
    expected_files = [
        os.path.join(temp_dir, "noria_optimization_history.csv"),
        os.path.join(temp_dir, "noria_objective_history.png"),
    ]
    for f in expected_files:
        assert os.path.exists(f), f"Expected file {f} was not generated."

def test_noria_optimization_history_format(run_noria_optimization_app):
    """
    Tests the format and basic content of the noria_optimization_history.csv file.
    """
    temp_dir = run_noria_optimization_app
    df = pd.read_csv(os.path.join(temp_dir, "noria_optimization_history.csv"))
    assert not df.empty, "Optimization history CSV is empty."
    assert all(col in df.columns for col in ['Q_in', 'k_q', 'objective'])
    assert len(df) > 1, "Optimization history should contain multiple entries."
