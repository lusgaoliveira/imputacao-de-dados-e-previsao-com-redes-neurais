import numpy as np

def mase(y_true, y_pred, y_train):
    # erro do modelo
    mae_model = np.mean(np.abs(y_true - y_pred))

    # erro do método naive no treino (X_{t} - X_{t-1})
    naive_mae = np.mean(np.abs(y_train[1:] - y_train[:-1]))

    return mae_model / naive_mae

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def mpe(y_true, y_pred):
    return np.mean((y_true - y_pred) / y_true) * 100

def seasonal_mase(y_true, y_pred, y, seasonality=365):
    num = np.mean(np.abs(y_true - y_pred))
    denom = np.mean(np.abs(y[seasonality:] - y[:-seasonality]))
    return num / denom