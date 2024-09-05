import pandas as pd
import numpy as np
import tensorflow as tf
import json

def load_parameters(config_path='../config/parameters.json'):
    with open(config_path, 'r') as file:
        params = json.load(file)
    return params

def backtest(data_path='../data/historical_data.csv', model_path='../models/crypto_trading_model.h5', config_path='../config/parameters.json'):
    params = load_parameters(config_path)
    initial_balance = params['initial_balance']
    stop_loss = params['stop_loss']
    take_profit = params['take_profit']
    transaction_cost = 0.001  # 0.1% transaction fee

    model = tf.keras.models.load_model(model_path)
    data = pd.read_csv(data_path)
    X = data[['open', 'high', 'low', 'close', 'volume']].values
    y = data['close'].values

    X = (X - X.mean(axis=0)) / X.std(axis=0)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    predictions = model.predict(X_test)

    balance = initial_balance
    btc_balance = 0
    for i in range(len(predictions) - 1):
        if predictions[i + 1] > predictions[i]:  # Buy signal
            btc_balance += (balance / y_test[i]) * (1 - transaction_cost)
            balance = 0
        elif predictions[i + 1] < predictions[i]:  # Sell signal
            balance += (btc_balance * y_test[i]) * (1 - transaction_cost)
            btc_balance = 0
        
        # Implement stop-loss
        if btc_balance > 0 and (y_test[i] < (1 - stop_loss) * y_test[i - 1]):
            balance += (btc_balance * y_test[i]) * (1 - transaction_cost)
            btc_balance = 0
        
        # Implement take-profit
        if btc_balance > 0 and (y_test[i] > (1 + take_profit) * y_test[i - 1]):
            balance += (btc_balance * y_test[i]) * (1 - transaction_cost)
            btc_balance = 0

    final_balance = balance + btc_balance * y_test[-1]
    print(f"Initial Balance: ${initial_balance}")
    print(f"Final Balance: ${final_balance}")
    print(f"Profit: ${final_balance - initial_balance}")

if __name__ == "__main__":
    backtest()
