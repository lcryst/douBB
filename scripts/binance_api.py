from dotenv import load_dotenv
import os
import ccxt

load_dotenv
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
})