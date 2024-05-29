import pickle
import pandas as pd
from pathlib import Path
    
def load(file_name, file_type):
    # Get the path to the current script
    current_dir = Path(__file__).resolve().parent

    # Navigate to the project root directory
    project_root = current_dir.parent.parent
    
    if file_type == 'pkl':
        # Construct the path to the file
        model_path = project_root / 'models' / file_name
        
        # Open the file in binary read mode and load the model
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)
        
        # Return the loaded model
        return loaded_model
    
    elif file_type == 'csv':
        # Construct the path to the file
        csv_path = project_root / 'data' / file_name
        
        # Load the data frame
        df = pd.read_csv(csv_path)

        # Return dataframe
        return df



    
    