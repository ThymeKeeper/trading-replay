#!/usr/bin/env python3
"""
Data downloader for Trading Replay application
Supports multiple data sources for downloading historical 5-minute bars
"""

import sys
import json
import pandas as pd
from datetime import datetime, timedelta
import argparse

def download_yfinance_data(symbol, period="2y", interval="5m"):
    """Download data using yfinance (Yahoo Finance)"""
    try:
        import yfinance as yf
    except ImportError:
        print("Please install yfinance: pip install yfinance")
        sys.exit(1)
    
    print(f"Downloading {symbol} from Yahoo Finance...")
    ticker = yf.Ticker(symbol)
    
    # For forex pairs, use the Yahoo Finance format
    if "/" in symbol:
        symbol = symbol.replace("/", "") + "=X"
        ticker = yf.Ticker(symbol)
    
    # Download data
    data = ticker.history(period=period, interval=interval)
    
    if data.empty:
        print(f"No data found for {symbol}")
        return None
    
    # Reset index to get timestamp as column
    data = data.reset_index()
    
    # Rename columns to match our format
    data = data.rename(columns={
        'Date': 'time',
        'Datetime': 'time',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    # Ensure time column exists
    if 'time' not in data.columns and data.index.name in ['Date', 'Datetime']:
        data = data.reset_index()
        data = data.rename(columns={data.columns[0]: 'time'})
    
    # Select only needed columns
    columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    data = data[columns]
    
    print(f"Downloaded {len(data)} bars from {data['time'].min()} to {data['time'].max()}")
    return data

def download_alphavantage_data(symbol, api_key, outputsize='full'):
    """Download data using Alpha Vantage API"""
    try:
        import requests
    except ImportError:
        print("Please install requests: pip install requests")
        sys.exit(1)
    
    print(f"Downloading {symbol} from Alpha Vantage...")
    
    # Determine function based on symbol type
    if "/" in symbol:  # Forex pair
        from_symbol, to_symbol = symbol.split("/")
        url = f"https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={from_symbol}&to_symbol={to_symbol}&interval=5min&apikey={api_key}&outputsize={outputsize}&datatype=csv"
    else:  # Stock
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={api_key}&outputsize={outputsize}&datatype=csv"
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error downloading data: {response.status_code}")
        return None
    
    # Parse CSV data
    from io import StringIO
    data = pd.read_csv(StringIO(response.text))
    
    if data.empty or 'timestamp' not in data.columns:
        print(f"No data found for {symbol}")
        print("Response:", response.text[:200])
        return None
    
    # Rename columns
    data = data.rename(columns={
        'timestamp': 'time',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    })
    
    # Sort by time (Alpha Vantage returns newest first)
    data['time'] = pd.to_datetime(data['time'])
    data = data.sort_values('time')
    
    print(f"Downloaded {len(data)} bars from {data['time'].min()} to {data['time'].max()}")
    return data

def save_to_csv(data, filename):
    """Save DataFrame to CSV"""
    data.to_csv(filename, index=False)
    print(f"Saved to {filename}")

def save_to_json(data, filename):
    """Save DataFrame to JSON in the format expected by the Rust app"""
    records = []
    for _, row in data.iterrows():
        # Convert timestamp to ISO format
        if isinstance(row['time'], str):
            timestamp = row['time']
        else:
            timestamp = row['time'].isoformat() + 'Z'
        
        records.append({
            'timestamp': timestamp,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row.get('volume', 0))
        })
    
    with open(filename, 'w') as f:
        json.dump(records, f, indent=2)
    
    print(f"Saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Download historical 5-minute bar data')
    parser.add_argument('symbol', help='Symbol to download (e.g., AAPL, EUR/USD)')
    parser.add_argument('-s', '--source', choices=['yfinance', 'alphavantage'], 
                       default='yfinance', help='Data source')
    parser.add_argument('-p', '--period', default='2y', 
                       help='Period for yfinance (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max)')
    parser.add_argument('-k', '--api-key', help='API key for Alpha Vantage')
    parser.add_argument('-f', '--format', choices=['csv', 'json'], 
                       default='csv', help='Output format')
    parser.add_argument('-o', '--output', help='Output filename (default: SYMBOL_5m.FORMAT)')
    
    args = parser.parse_args()
    
    # Generate output filename if not specified
    if not args.output:
        symbol_clean = args.symbol.replace('/', '_').replace('=', '')
        args.output = f"{symbol_clean}_5m.{args.format}"
    
    # Download data
    data = None
    if args.source == 'yfinance':
        data = download_yfinance_data(args.symbol, args.period)
    elif args.source == 'alphavantage':
        if not args.api_key:
            print("Error: Alpha Vantage requires an API key. Get one free at https://www.alphavantage.co/support/#api-key")
            sys.exit(1)
        data = download_alphavantage_data(args.symbol, args.api_key)
    
    if data is None:
        print("Failed to download data")
        sys.exit(1)
    
    # Save data
    if args.format == 'csv':
        save_to_csv(data, args.output)
    else:
        save_to_json(data, args.output)

if __name__ == '__main__':
    main()