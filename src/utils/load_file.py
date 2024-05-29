import pickle
import pandas as pd
from pathlib import Path

def load(file_name, file_type):
    """
    Load a file of specified type.

    Parameters:
        file_name (str): The name of the file to load.
        file_type (str): The type of the file ('pkl' for pickle or 'csv' for CSV).

    Returns:
        loaded_object: The loaded object (model or DataFrame).
    """
    try:
        # Get the path to the current script
        current_dir = Path(__file__).resolve().parent

        # Navigate to the project root directory
        project_root = current_dir.parent.parent
        
        if file_type == 'pkl':
            # Construct the path to the file
            model_path = project_root / 'models' / file_name
            
            # Open the file in binary read mode and load the model
            with open(model_path, 'rb') as file:
                loaded_object = pickle.load(file)
        
        elif file_type == 'csv':
            # Construct the path to the file
            csv_path = project_root / 'data' / file_name
            
            # Load the data frame
            loaded_object = pd.read_csv(csv_path)
        
        else:
            raise ValueError("Invalid file type. Only 'pkl' and 'csv' are supported.")

        return loaded_object
    
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None