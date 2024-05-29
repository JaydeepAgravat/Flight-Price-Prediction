import pandas as pd
import numpy as np
from src.utils.load_file import load

MODEL_NAME = 'RandomForestRegressor.pkl'

def prediction(query):
    """
    Make a prediction using a pre-trained model.

    Parameters:
        query (pd.DataFrame or np.ndarray): The input data for prediction.

    Returns:
        float: The predicted value.
    """
    try:
        # Load the pre-trained model from a pickle file
        model = load(MODEL_NAME, 'pkl')
        
        # Ensure the input query is in the correct format for prediction
        if isinstance(query, pd.DataFrame) or isinstance(query, np.ndarray):
            # Make a prediction using the loaded model
            result = model.predict(query)[0]
            return result
        else:
            raise ValueError("Query must be a DataFrame or ndarray.")
    
    except FileNotFoundError:
        print(f"Model file '{MODEL_NAME}' not found.")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
