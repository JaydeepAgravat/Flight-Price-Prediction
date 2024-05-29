from pathlib import Path
import pandas as pd

def save(file_name, X, y):
    # Get the path to the current script
    current_dir = Path(__file__).resolve().parent

    # Navigate to the project root directory
    project_root = current_dir.parent.parent

    # Construct the path to the file
    config_file = project_root / 'data' / file_name

    # Save dataframe
    X.join(y).to_csv(config_file, index=False)