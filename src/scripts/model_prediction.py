import pandas as pd
from src.utils.load_file import load

MODEL_NAME = 'RandomForestRegressor.pkl'

def prediction(query):
    model = load(MODEL_NAME, 'pkl')
    return model.predict(query)[0]
