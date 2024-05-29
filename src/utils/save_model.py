from pathlib import Path
import pickle

def save(model, model_name):
    """
    Save a model to a pickle file.

    Parameters:
        model (object): The model to save.
        model_name (str): The name of the file to save the model in (without extension).

    Returns:
        None
    """
    try:
        # Get the path to the current script
        current_dir = Path(__file__).resolve().parent

        # Navigate to the project root directory
        project_root = current_dir.parent.parent

        # Construct the path to the file
        model_file = project_root / 'models' / f'{model_name}.pkl'

        # Save the model to a pickle file
        with open(model_file, 'wb') as file:
            pickle.dump(model, file)

        print(f"Model '{model_name}.pkl' saved successfully.")

    except FileNotFoundError:
        print(f"Directory not found. Failed to save the model '{model_name}.pkl'.")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")
