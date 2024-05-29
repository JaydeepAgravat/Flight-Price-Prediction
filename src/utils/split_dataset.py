from save_dataset import save
import pandas as pd
from sklearn.model_selection import train_test_split

def split(df, type=None):
    """
    Split the DataFrame into training, validation, and test sets.

    Parameters:
        df (pd.DataFrame): The DataFrame to split.
        type (str, optional): The type of split to return ('train', 'validation', 'test'). 
                              If None, all splits are returned.

    Returns:
        tuple: Depending on the type parameter, returns the corresponding split(s).
    """
    try:
        # Separate features and target variable
        X = df.drop(columns="price")
        y = df.price.copy()

        # First split to create test set
        X_, X_test, y_, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Second split to create training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=0.2, random_state=42)
        
        if type == 'train':
            return X_train, y_train
        elif type == 'validation':
            return X_val, y_val
        elif type == 'test':
            return X_test, y_test
        else:
            return X_train, y_train, X_val, y_val, X_test, y_test
    
    except KeyError as e:
        print(f"Column not found in DataFrame: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during the split: {e}")
        return None

def split_and_save(df):
    """
    Split the DataFrame into training, validation, and test sets, and save them to files.

    Parameters:
        df (pd.DataFrame): The DataFrame to split and save.

    Returns:
        None
    """
    try:
        # Perform the split
        splits = split(df)
        if splits is None:
            return

        X_train, y_train, X_val, y_val, X_test, y_test = splits
        
        # Save the splits to files
        save('train', X_train, y_train)
        save('validation', X_val, y_val)
        save('test', X_test, y_test)
        
    except Exception as e:
        print(f"An error occurred during splitting and saving: {e}")
