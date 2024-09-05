import pandas as pd
import json
from binance_api import exchange

def load_parameters(config_path='../config/parameters.json'):
    with open(config_path, 'r') as file:
        params = json.load(file)
    return params

def collect_data(config_path='../config/parameters.json'):
    params = load_parameters(config_path)
    symbol = params['currency_pair']
    timeframe = params['timeframe']
    start_date = params['start_date']
    end_date = params['end_date']

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=exchange.parse8601(start_date))
    data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data = data[data['timestamp'] <= pd.to_datetime(end_date)]
    data.to_csv('../data/historical_data.csv', index=False)
    print("Data collected and saved to historical_data.csv")

if __name__ == "__main__":
    collect_data()
