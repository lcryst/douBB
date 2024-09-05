import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import json

def load_parameters(config_path='../config/parameters.json'):
    with open(config_path, 'r') as file:
        params = json.load(file)
    return params

def build_model(learning_rate=0.001):
    model = Sequential()
    model.add(Dense(64, input_dim=5, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

def train_model(data_path='../data/historical_data.csv', model_path='../models/crypto_trading_model.h5', config_path='../config/parameters.json'):
    params = load_parameters(config_path)
    data = pd.read_csv(data_path)
    X = data[['open', 'high', 'low', 'close', 'volume']].values
    y = data['close'].values

    X = (X - X.mean(axis=0)) / X.std(axis=0)

    model = KerasRegressor(build_fn=build_model, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'epochs': [50, 100],
        'batch_size': [32, 64]
    }
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_result = grid.fit(X, y)

    best_model = grid_result.best