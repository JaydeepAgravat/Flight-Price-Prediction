import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def evaluation(model, type, X, y):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    n = len(y)
    p = X.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f'{type} R^2: {r2:.4f}')
    print(f'{type} Adjusted R^2: {adj_r2:.4f}')
    print(f'{type} RMSE: {rmse:.4f}')