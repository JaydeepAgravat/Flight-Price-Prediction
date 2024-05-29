from pathlib import Path
import pandas as pd

def save(file_name, X, y):
    """
    Save a DataFrame and target series to a CSV file.

    Parameters:
        file_name (str): The name of the file to save.
        X (pd.DataFrame): The features DataFrame.
        y (pd.Series or pd.DataFrame): The target series or DataFrame.

    Returns:
        None
    """
    try:
        # Get the path to the current script
        current_dir = Path(__file__).resolve().parent

        # Navigate to the project root directory
        project_root = current_dir.parent.parent

        # Construct the path to the file
        config_file = project_root / 'data' / file_name

        # Ensure that 'y' is a DataFrame if it's a Series
        if isinstance(y, pd.Series):
            y = y.to_frame()

        # Save the DataFrame to a CSV file
        X.join(y).to_csv(config_file, index=False)
        print(f"File '{file_name}' saved successfully.")

    except FileNotFoundError:
        print(f"Directory not found. Failed to save the file '{file_name}'.")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")
