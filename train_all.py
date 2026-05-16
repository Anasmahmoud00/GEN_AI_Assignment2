"""
train_all.py
------------
Smart training script that detects the correct directory structure.
"""

import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models = ["mlp", "gan", "vae", "diffusion"]


def train():
    for model_name in models:
        print("\n" + "=" * 50)
        print(" Starting Training for: " + model_name.upper())
        print("=" * 50 + "\n")
        
        script_path = os.path.join(BASE_DIR, "model", model_name, "train.py")
        
        path_exists = os.path.exists(script_path)
        if not path_exists:
            print("[ERROR] Could not find script at: " + script_path)
            cwd_path = os.getcwd()
            print("Current Working Directory: " + cwd_path)
            continue

        python_executable = sys.executable
        process_args = [python_executable, script_path]
        
        result = subprocess.run(process_args, cwd=BASE_DIR)
        
        exit_code = result.returncode
        if exit_code == 0:
            print("\n[SUCCESS] " + model_name.upper() + " training completed.")
        else:
            print("\n[FAILED] " + model_name.upper() + " training failed.")


if __name__ == "__main__":
    train()

