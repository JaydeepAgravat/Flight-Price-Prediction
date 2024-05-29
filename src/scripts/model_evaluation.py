import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def evaluation(model, type, X, y):
    """
    Evaluate a model's performance on a given dataset.

    Parameters:
        model (object): The model to evaluate.
        type (str): The type of dataset (e.g., 'train', 'validation', 'test').
        X (pd.DataFrame or np.ndarray): The features of the dataset.
        y (pd.Series or np.ndarray): The target variable of the dataset.

    Returns:
        None
    """
    try:
        # Predict the target variable using the model
        y_pred = model.predict(X)
        
        # Calculate R-squared
        r2 = r2_score(y, y_pred)
        
        # Calculate adjusted R-squared
        n = len(y)
        p = X.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        # Calculate Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Print evaluation metrics
        print(f'{type} R^2: {r2:.4f}')
        print(f'{type} Adjusted R^2: {adj_r2:.4f}')
        print(f'{type} RMSE: {rmse:.4f}')
    
    except AttributeError as e:
        print(f"Attribute error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
