
import pytest
import subprocess
import os
import pandas as pd
import numpy as np

# Define the path to the main application script
MAIN_APP_PATH = "src/main.py"
BASELINE_DATA_DIR = "tests/baseline_data"

@pytest.fixture(scope="module")
def run_main_app_for_regression():
    """
    Runs the main application script as a subprocess for regression testing.
    """
    # Run the main application script
    try:
        result = subprocess.run(
            ["python", MAIN_APP_PATH, "--inference-only", "--snapshot-dir", "tests/baseline_data/tank_model_snapshot"],
            capture_output=True, text=True, check=True
        )
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running main.py: {e}")
        print(e.stdout)
        print(e.stderr)
        pytest.fail(f"main.py failed with exit code {e.returncode}")

    yield

    # Teardown: Clean up generated files
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

def test_inference_csv_against_baseline(run_main_app_for_regression):
    """
    Compares the generated inference_data.csv against the baseline version.
    """
    baseline_csv = os.path.join(BASELINE_DATA_DIR, "inference_data.csv")
    generated_csv = "inference_data.csv"
    
    assert os.path.exists(generated_csv), "inference_data.csv was not generated."
    
    # Use pandas to read and compare the CSV files, allowing for minor floating point differences
    df_baseline = pd.read_csv(baseline_csv)
    df_generated = pd.read_csv(generated_csv)
    
    # Compare numerical columns with numpy.allclose for better floating point tolerance
    # Adjust rtol and atol as needed based on expected variations
    for col in df_baseline.columns:
        if df_baseline[col].dtype == 'float64' or df_baseline[col].dtype == 'float32':
            assert np.allclose(df_baseline[col].values, df_generated[col].values, rtol=1e-3, atol=1e-4), f"Column {col} differs significantly."
        else:
            pd.testing.assert_series_equal(df_baseline[col], df_generated[col])
