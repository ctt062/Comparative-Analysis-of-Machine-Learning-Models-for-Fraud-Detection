# main.py
import os
import subprocess
import sys

def run_script(script_name):
    """Executes a Python script and checks for errors."""
    print(f"\n--- Running {script_name} ---")
    try:
        # Using sys.executable to ensure the same python interpreter is used
        result = subprocess.run([sys.executable, script_name], check=True, text=True, capture_output=True)
        print(result.stdout)
        print(f"--- Finished {script_name} successfully ---\n")
    except subprocess.CalledProcessError as e:
        print(f"--- ERROR in {script_name} ---")
        print(e.stdout)
        print(e.stderr)
        sys.exit(f"Script {script_name} failed. Exiting pipeline.")

# --- Setup ---
print("Setting up project directories...")
# Create necessary directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('graph', exist_ok=True)
os.makedirs('code', exist_ok=True)

# Check for the dataset
data_path = 'code/onlinefraud.csv'
if not os.path.exists(data_path):
    print("\n" + "="*50)
    print(f"ERROR: Dataset not found at '{data_path}'")
    print("Please download the dataset and place 'onlinefraud.csv' in the 'code' folder.")
    print("You can download it from: https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection")
    print("="*50 + "\n")
    sys.exit(1)

# --- Pipeline Execution ---
print("Executing the fraud detection pipeline...")

scripts_to_run = [
    "code/01_data_cleaning_eda.py",
    "code/02_data_preprocessing.py",
    "code/03_model_training_evaluation.py"
]

for script in scripts_to_run:
    run_script(script)

print("✅ Pipeline execution complete.")
print("Check the 'graph' folder for visualizations and the console output for results.")