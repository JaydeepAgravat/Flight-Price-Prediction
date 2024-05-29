from .save_dataset import save
import pandas as pd
from sklearn.model_selection import train_test_split

def split(df, type=None):
    X = df.drop(columns="price")
    y = df.price.copy()

    X_, X_test, y_, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=0.2, random_state=42)
    
    if type == 'train': return X_train, y_train
    elif type == 'validation': return X_val, y_val
    elif type == 'test': return X_test, y_test
    else: return X_train, y_train, X_val, y_val, X_test, y_test
    
def split_and_save(file_name):
    X_train, y_train, X_val, y_val, X_test, y_test = split(file_name)
    save('train', X_train, y_train)
    save('validation', X_val, y_val)
    save('test', X_test, y_test)