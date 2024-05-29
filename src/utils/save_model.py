from pathlib import Path
import pickle

def save(model, model_name):
    # Get the path to the current script
    current_dir = Path(__file__).resolve().parent

    # Navigate to the project root directory
    project_root = current_dir.parent.parent

    # Construct the path to the file
    model_file = project_root / 'models' / model_name

    # Save the model
    with open(f'{model_file}.pkl', 'wb') as file:
        pickle.dump(model, file)
        
    