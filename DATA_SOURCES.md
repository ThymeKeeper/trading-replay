# Data Sources for Trading Replay

This application can load historical 5-minute bar data from various sources. Here's how to get data:

## 1. TradingView Export (Recommended for Forex)

1. Open TradingView and navigate to your desired chart
2. Set timeframe to **5 minutes**
3. Scroll left to load historical data (can get several months)
4. Press **Alt+Shift+E** or click "Export chart data"
5. Save the CSV file

## 2. Yahoo Finance (Using provided script)

Great for stocks and major forex pairs:

```bash
# Install dependencies
pip install yfinance pandas

# Download EUR/USD data (2 years)
python download_data.py EUR/USD

# Download Apple stock (1 year)
python download_data.py AAPL -p 1y

# Save as JSON instead of CSV
python download_data.py AAPL -f json
```

## 3. Alpha Vantage API

Free tier available (5 API calls/minute, 500 calls/day):

```bash
# Get free API key from: https://www.alphavantage.co/support/#api-key

# Download forex data
python download_data.py EUR/USD -s alphavantage -k YOUR_API_KEY

# Download stock data
python download_data.py MSFT -s alphavantage -k YOUR_API_KEY
```

## 4. Questrade API

For Questrade users, you can export data via their API and save as CSV:

```
time,open,high,low,close,volume
2024-01-01 09:30:00,1.0850,1.0855,1.0848,1.0852,1500
2024-01-01 09:35:00,1.0852,1.0858,1.0851,1.0856,1200
```

## CSV Format Requirements

The CSV file should have headers and contain:
- `time` or `timestamp`: Date/time in any common format
- `open`, `high`, `low`, `close`: Price data
- `volume` (optional): Trading volume

Supported timestamp formats:
- Unix timestamp (seconds or milliseconds)
- ISO 8601: `2024-01-01T09:30:00Z`
- Common formats: `2024-01-01 09:30:00`, `01/01/2024 09:30`

## Loading Data in the Application

1. Click the **📁 Load Data** button
2. Select your CSV or JSON file
3. The instrument name will be extracted from the filename
4. Data will be loaded and you'll see the date range

## Tips

- For best results, load at least 1000 bars (3-4 days of data)
- More history allows better analysis of patterns
- The application starts at bar 60 to ensure indicators have enough data
- Non-trading hours (weekends) will be automatically dimmed

## Sample Data Download Commands

```bash
# Forex pairs (2 years of data)
python download_data.py EUR/USD
python download_data.py GBP/USD
python download_data.py USD/JPY

# Stocks (1 year of data)
python download_data.py SPY -p 1y
python download_data.py AAPL -p 1y
python download_data.py TSLA -p 1y

# Crypto (requires specific format on Yahoo)
python download_data.py BTC-USD -p 3mo
python download_data.py ETH-USD -p 3mo
```