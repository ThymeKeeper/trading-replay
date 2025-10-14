use chrono::{DateTime, Utc, NaiveDateTime, TimeZone};
use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use crate::Bar;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CsvRecord {
    #[serde(rename = "time")]
    timestamp: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct TradingViewCsvRecord {
    time: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    #[serde(default)]
    volume: Option<f64>,
}

pub fn load_bars_from_csv(path: &Path) -> Result<Vec<Bar>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = csv::Reader::from_reader(file);
    
    let mut bars = Vec::new();
    
    // Try to detect the format by reading the headers
    let headers = reader.headers()?.clone();
    let has_volume = headers.iter().any(|h| h.to_lowercase() == "volume");
    
    // Reset reader
    drop(reader);
    let file = File::open(path)?;
    let mut reader = csv::Reader::from_reader(file);
    
    for result in reader.records() {
        let record = result?;
        
        // Parse timestamp - handle various formats
        let timestamp_str = &record[0];
        let timestamp = parse_timestamp(timestamp_str)?;
        
        // Parse OHLC data
        let open: f64 = record[1].parse()?;
        let high: f64 = record[2].parse()?;
        let low: f64 = record[3].parse()?;
        let close: f64 = record[4].parse()?;
        
        // Parse volume if available
        let volume = if has_volume && record.len() > 5 {
            record[5].parse().unwrap_or(0.0)
        } else {
            0.0
        };
        
        bars.push(Bar {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        });
    }
    
    // Sort bars by timestamp to ensure chronological order
    bars.sort_by_key(|bar| bar.timestamp);
    
    Ok(bars)
}

fn parse_timestamp(timestamp_str: &str) -> Result<DateTime<Utc>, Box<dyn Error>> {
    // Try different timestamp formats
    
    // Unix timestamp (seconds)
    if let Ok(unix_ts) = timestamp_str.parse::<i64>() {
        if unix_ts > 1_000_000_000 && unix_ts < 10_000_000_000 {
            return Ok(Utc.timestamp_opt(unix_ts, 0).unwrap());
        }
        // Unix timestamp (milliseconds)
        if unix_ts > 1_000_000_000_000 {
            return Ok(Utc.timestamp_opt(unix_ts / 1000, 0).unwrap());
        }
    }
    
    // ISO 8601 format
    if let Ok(dt) = DateTime::parse_from_rfc3339(timestamp_str) {
        return Ok(dt.with_timezone(&Utc));
    }
    
    // Common date formats
    let formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    ];
    
    for format in &formats {
        if let Ok(naive_dt) = NaiveDateTime::parse_from_str(timestamp_str, format) {
            return Ok(Utc.from_utc_datetime(&naive_dt));
        }
    }
    
    Err(format!("Unable to parse timestamp: {}", timestamp_str).into())
}

pub fn load_bars_from_json(path: &Path) -> Result<Vec<Bar>, Box<dyn Error>> {
    let file = File::open(path)?;
    let bars: Vec<Bar> = serde_json::from_reader(file)?;
    Ok(bars)
}

#[allow(dead_code)]
pub fn save_bars_to_json(bars: &[Bar], path: &Path) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, bars)?;
    Ok(())
}

// Helper function to export TradingView data
pub fn create_tradingview_export_instructions() -> String {
    r#"
How to Export Data from TradingView:

1. Open TradingView and navigate to your desired chart
2. Set the timeframe to 5 minutes
3. Load as much historical data as possible by scrolling left
4. Click on the "Export chart data" button (or use Alt+Shift+E)
5. Save the CSV file

The CSV should have columns: time, open, high, low, close, volume

Alternative Data Sources:

1. Questrade API:
   - Use the Questrade API to fetch historical data
   - Export to CSV format with timestamp, OHLC, volume

2. Yahoo Finance (via Python):
   ```python
   import yfinance as yf
   import pandas as pd
   
   ticker = yf.Ticker("EURUSD=X")
   data = ticker.history(period="2y", interval="5m")
   data.to_csv("EURUSD_5m.csv")
   ```

3. Alpha Vantage API:
   - Free tier available
   - Provides forex and stock data
   - CSV export available

Supported CSV formats:
- time/timestamp, open, high, low, close, volume
- Unix timestamps or common date formats
- Headers are optional but recommended
"#.to_string()
}