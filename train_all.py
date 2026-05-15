"""
train_all.py
------------
Smart training script that detects the correct directory structure.
"""

import subprocess
import sys
import os

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models = ["mlp", "gan", "vae", "flow"]

def train():
    for model in models:
        print(f"\n" + "="*50)
        print(f" Starting Training for: {model.upper()} ")
        print("="*50 + "\n")
        
        # Build path relative to the script location
        script_path = os.path.join(BASE_DIR, "model", model, "train.py")
        
        if not os.path.exists(script_path):
            print(f"[ERROR] Could not find script at: {script_path}")
            print(f"Current Working Directory: {os.getcwd()}")
            continue

        # Run the training script
        # Setting the CWD to BASE_DIR ensures internal imports work
        result = subprocess.run([sys.executable, script_path], cwd=BASE_DIR)
        
        if result.returncode == 0:
            print(f"\n[SUCCESS] {model.upper()} training completed.")
        else:
            print(f"\n[FAILED] {model.upper()} training failed.")

if __name__ == "__main__":
    train()
