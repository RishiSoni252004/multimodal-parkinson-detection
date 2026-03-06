import sys
import os

# Ensure the root directory is in the path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.train import train_and_compare_models

if __name__ == "__main__":
    print("Starting Training Pipeline...")
    train_and_compare_models()
    print("Training Pipeline Completed Successfully.")
