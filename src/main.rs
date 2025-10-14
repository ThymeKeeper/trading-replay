use eframe::egui;
use chrono::{DateTime, Utc, Datelike, Timelike};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

mod data_loader;
use data_loader::{load_bars_from_csv, load_bars_from_json};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Bar {
    timestamp: DateTime<Utc>,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Transaction {
    timestamp: DateTime<Utc>,
    instrument: String,
    action: TradeAction,
    lots: i32,
    price: f64,
    position_after: i32,
    trade_pnl: f64,
    gross_pnl: f64, // P&L before slippage
    cumulative_pnl: f64,
    max_unrealized_drawdown: f64,
    equity_tied_up: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum TradeAction {
    Buy,
    Sell,
    Flatten,
}

struct TradingReplayApp {
    instrument: String,
    bars: Vec<Bar>,
    current_bar_index: usize,
    position: i32,
    lot_size: i32,
    lot_size_string: String,
    transactions: Vec<Transaction>,
    playing: bool,
    speed: f32,
    last_update: std::time::Instant,
    average_entry_price: f64,
    cumulative_pnl: f64,
    t3_values: Vec<f64>,
    t3_fast_values: Vec<f64>, // T3 with period 2 for MACD
    t3_slow_values: Vec<f64>, // T3 with period 10 for MACD
    macd_values: Vec<f64>,    // MACD line (T3_fast - T3_slow)
    macd_signal: Vec<f64>,    // Signal line (9-EMA of MACD)
    macd_histogram: Vec<f64>, // Histogram (MACD - Signal)
    average_price_history: Vec<(usize, f64)>, // (bar_index, avg_price)
    show_data_menu: bool,
    data_path: Option<PathBuf>,
    load_error: Option<String>,
    view_offset: i32, // Offset from current bar for scrolling
    max_unrealized_drawdown: f64, // Track worst unrealized P&L for current position
    position_high_water_mark: f64, // Track best unrealized P&L for current position
    zoom_level: f32, // Zoom factor for bar width (1.0 = default)
    closed_trades: Vec<ClosedTrade>, // Track closed trades for visualization
    position_entry_bar: usize, // Track when position was opened
    start_date_input: String, // User input for start date
    auto_flatten_on_profit: bool, // Auto-flatten when position becomes profitable
    disable_when_positioned: bool, // Disable buy/sell when already in position
    auto_flatten_at_close: bool, // Auto-flatten positions at market close
    slippage_per_lot: f64, // Slippage/spread per lot in price units
    display_timezone: chrono::FixedOffset, // Timezone for displaying times
    t3_fast_length: usize,
    t3_slow_length: usize,
    t3_fast_length_string: String,
    t3_slow_length_string: String,
    stop_loss_dollars: Option<f64>,
    take_profit_dollars: Option<f64>,
    stop_loss_string: String,
    take_profit_string: String,
    show_macd: bool,
    delay_entry_consecutive_bars: bool,
    pending_buy_order: bool,
    pending_sell_order: bool,
    show_linreg: bool,
    linreg_period: usize,
    // Market regime detection
    volatility_20: Vec<f64>,
    current_regime: MarketRegime,
    regime_threshold: f64, // Volatility threshold for regime change
    show_regime_indicator: bool,
    // Regime-based strategy settings
    low_vol_strategy: TradingStrategy,
    high_vol_strategy: TradingStrategy,
    enable_strategy_signals: bool,
    last_strategy_signal: Option<bool>, // true = buy signal, false = sell signal
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum MarketRegime {
    LowVolatility,
    HighVolatility,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TradingStrategy {
    Momentum,    // Trade with the trend
    MeanRevert,  // Trade reversals
    NoTrade,     // Don't trade in this regime
}

#[derive(Debug, Clone)]
struct ClosedTrade {
    entry_bar_index: usize,
    exit_bar_index: usize,
    entry_price: f64,
    exit_price: f64,
    pnl: f64,
    was_long: bool,
    price_history: Vec<(usize, f64)>, // Track all average price changes during the trade
}

impl Default for TradingReplayApp {
    fn default() -> Self {
        let bars = generate_sample_bars();
        let start_index = Self::get_random_start_index(bars.len());
        
        let mut app = Self {
            instrument: "EUR/USD".to_string(),
            bars: bars.clone(),
            current_bar_index: start_index,
            position: 0,
            lot_size: 1,
            lot_size_string: "1".to_string(),
            transactions: Vec::new(),
            playing: false,
            speed: 1.0,
            last_update: std::time::Instant::now(),
            average_entry_price: 0.0,
            cumulative_pnl: 0.0,
            t3_values: Vec::new(),
            t3_fast_values: Vec::new(),
            t3_slow_values: Vec::new(),
            macd_values: Vec::new(),
            macd_signal: Vec::new(),
            macd_histogram: Vec::new(),
            average_price_history: Vec::new(),
            show_data_menu: false,
            data_path: None,
            load_error: None,
            view_offset: 10, // Start with buffer visible
            max_unrealized_drawdown: 0.0,
            position_high_water_mark: 0.0,
            zoom_level: 1.0,
            closed_trades: Vec::new(),
            position_entry_bar: start_index,
            start_date_input: String::new(),
            auto_flatten_on_profit: false,
            disable_when_positioned: false,
            auto_flatten_at_close: false,
            slippage_per_lot: 0.0002, // Default to 2 pips for forex, $0.0002 for stocks
            display_timezone: chrono::FixedOffset::west_opt(5 * 3600).unwrap(), // EST by default
            t3_fast_length: 2,
            t3_slow_length: 10,
            t3_fast_length_string: "2".to_string(),
            t3_slow_length_string: "10".to_string(),
            stop_loss_dollars: None,
            take_profit_dollars: None,
            stop_loss_string: String::new(),
            take_profit_string: String::new(),
            show_macd: true,  // Default to showing MACD
            delay_entry_consecutive_bars: false,
            pending_buy_order: false,
            pending_sell_order: false,
            show_linreg: false,
            linreg_period: 20,
            volatility_20: Vec::new(),
            current_regime: MarketRegime::LowVolatility,
            regime_threshold: 5.0, // 5% annualized volatility threshold
            show_regime_indicator: true,
            low_vol_strategy: TradingStrategy::Momentum,
            high_vol_strategy: TradingStrategy::MeanRevert,
            enable_strategy_signals: false,
            last_strategy_signal: None,
        };
        
        // Calculate T3 values up to starting position for better performance
        app.calculate_t3_values_up_to(start_index);
        app
    }
}

impl TradingReplayApp {
    fn get_random_start_index(total_bars: usize) -> usize {
        use std::time::{SystemTime, UNIX_EPOCH};
        
        // Minimum bars needed: 200 for preload + some for replay
        let min_start = 200;
        let max_start = total_bars.saturating_sub(100); // Leave at least 100 bars for replay
        
        if min_start >= max_start {
            return min_start.min(total_bars.saturating_sub(1));
        }
        
        // Use system time as seed for randomization
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as usize;
        
        // Random value between min_start and max_start
        min_start + (seed % (max_start - min_start))
    }
    
    fn find_bar_by_date(&self, date_str: &str) -> Option<usize> {
        use chrono::NaiveDate;
        
        // Parse the date string
        if let Ok(target_date) = NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
            // Find the first bar on or after this date
            for (idx, bar) in self.bars.iter().enumerate() {
                if bar.timestamp.date_naive() >= target_date {
                    // Ensure we have enough preload bars
                    return Some(idx.max(200));
                }
            }
        }
        None
    }
    
    fn jump_to_date(&mut self) {
        if let Some(index) = self.find_bar_by_date(&self.start_date_input) {
            self.current_bar_index = index;
            self.playing = false;
            
            // Clear trading state
            self.position = 0;
            self.average_entry_price = 0.0;
            self.cumulative_pnl = 0.0;
            self.transactions.clear();
            self.average_price_history.clear();
            self.closed_trades.clear();
            self.max_unrealized_drawdown = 0.0;
            self.position_high_water_mark = 0.0;
            
            // Recalculate T3 up to this point
            self.calculate_t3_values_up_to(self.current_bar_index);
        }
    }

    fn load_data_from_file(&mut self, path: PathBuf) {
        self.load_error = None;
        
        let result = if path.extension().and_then(|s| s.to_str()) == Some("json") {
            load_bars_from_json(&path)
        } else {
            load_bars_from_csv(&path)
        };
        
        match result {
            Ok(bars) if bars.len() > 300 => {  // Need at least 300 bars for proper preload
                self.bars = bars;
                self.data_path = Some(path.clone());
                
                // Extract instrument name from filename
                if let Some(stem) = path.file_stem() {
                    if let Some(name) = stem.to_str() {
                        self.instrument = name.to_string();
                    }
                }
                
                // Reset everything with randomized start
                self.current_bar_index = Self::get_random_start_index(self.bars.len());
                self.position = 0;
                self.average_entry_price = 0.0;
                self.cumulative_pnl = 0.0;
                self.transactions.clear();
                self.average_price_history.clear();
                self.calculate_t3_values_up_to(self.current_bar_index);
                self.playing = false;
                self.show_data_menu = false;
                self.view_offset = 10; // Start with buffer visible
                self.max_unrealized_drawdown = 0.0;
                self.position_high_water_mark = 0.0;
                self.closed_trades.clear();
            }
            Ok(_) => {
                self.load_error = Some("File must contain at least 300 bars for proper analysis".to_string());
            }
            Err(e) => {
                self.load_error = Some(format!("Error loading file: {}", e));
            }
        }
    }

    fn calculate_volatility(&self, end_bar_index: usize, period: usize) -> Option<f64> {
        if end_bar_index < period || period < 2 {
            return None;
        }
        
        // Calculate returns
        let mut returns = Vec::new();
        for i in (end_bar_index - period + 1)..=end_bar_index {
            if i > 0 {
                let return_val = (self.bars[i].close - self.bars[i-1].close) / self.bars[i-1].close;
                returns.push(return_val);
            }
        }
        
        if returns.is_empty() {
            return None;
        }
        
        // Calculate mean return
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        
        // Calculate standard deviation
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        
        let std_dev = variance.sqrt();
        
        // Annualize the volatility (5-minute bars: 78 per day, 252 trading days)
        let annualized_vol = std_dev * (78.0_f64 * 252.0_f64).sqrt();
        
        Some(annualized_vol * 100.0) // Return as percentage
    }
    
    fn update_market_regime(&mut self) {
        if self.current_bar_index >= 20 && self.volatility_20.len() > self.current_bar_index {
            let current_vol = self.volatility_20[self.current_bar_index];
            
            self.current_regime = if current_vol > self.regime_threshold {
                MarketRegime::HighVolatility
            } else {
                MarketRegime::LowVolatility
            };
        }
    }
    
    fn get_current_strategy(&self) -> TradingStrategy {
        match self.current_regime {
            MarketRegime::LowVolatility => self.low_vol_strategy,
            MarketRegime::HighVolatility => self.high_vol_strategy,
        }
    }
    
    fn is_trading_allowed(&self) -> bool {
        self.get_current_strategy() != TradingStrategy::NoTrade
    }
    
    fn get_strategy_signal(&self) -> Option<bool> {
        if !self.enable_strategy_signals || self.current_bar_index < 20 {
            return None;
        }
        
        let strategy = self.get_current_strategy();
        if strategy == TradingStrategy::NoTrade {
            return None;
        }
        
        match strategy {
            TradingStrategy::Momentum => self.get_momentum_signal(),
            TradingStrategy::MeanRevert => self.get_mean_revert_signal(),
            TradingStrategy::NoTrade => None,
        }
    }
    
    fn get_momentum_signal(&self) -> Option<bool> {
        // Momentum: Buy when price crosses above 20-bar SMA, sell when crosses below
        if self.current_bar_index < 20 {
            return None;
        }
        
        let current_price = self.bars[self.current_bar_index].close;
        let prev_price = if self.current_bar_index > 0 {
            self.bars[self.current_bar_index - 1].close
        } else {
            current_price
        };
        
        // Calculate 20-bar SMA
        let mut sum = 0.0;
        for i in (self.current_bar_index.saturating_sub(19))..=self.current_bar_index {
            sum += self.bars[i].close;
        }
        let sma = sum / 20.0;
        
        // Check for crossover
        if prev_price <= sma && current_price > sma {
            Some(true) // Buy signal
        } else if prev_price >= sma && current_price < sma {
            Some(false) // Sell signal
        } else {
            None
        }
    }
    
    fn get_mean_revert_signal(&self) -> Option<bool> {
        // Mean Revert: Buy when price is 2 std devs below 20-bar mean, sell when 2 std devs above
        if self.current_bar_index < 20 {
            return None;
        }
        
        let current_price = self.bars[self.current_bar_index].close;
        
        // Calculate 20-bar mean and std dev
        let mut sum = 0.0;
        for i in (self.current_bar_index.saturating_sub(19))..=self.current_bar_index {
            sum += self.bars[i].close;
        }
        let mean = sum / 20.0;
        
        let mut variance_sum = 0.0;
        for i in (self.current_bar_index.saturating_sub(19))..=self.current_bar_index {
            let diff = self.bars[i].close - mean;
            variance_sum += diff * diff;
        }
        let std_dev = (variance_sum / 20.0).sqrt();
        
        let upper_band = mean + 2.0 * std_dev;
        let lower_band = mean - 2.0 * std_dev;
        
        // Generate signals
        if current_price <= lower_band && self.position <= 0 {
            Some(true) // Buy signal (oversold)
        } else if current_price >= upper_band && self.position >= 0 {
            Some(false) // Sell signal (overbought)
        } else {
            None
        }
    }
    
    fn calculate_linear_regression(&self, end_bar_index: usize, period: usize) -> Option<(f64, f64, f64)> {
        if end_bar_index < period - 1 || end_bar_index >= self.bars.len() || period == 0 {
            return None;
        }
        
        let start_index = end_bar_index + 1 - period;
        
        // Calculate linear regression using least squares
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;
        
        for i in 0..period {
            let x = i as f64;
            let y = self.bars[start_index + i].close;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }
        
        let n = period as f64;
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;
        
        // Calculate standard deviation for channel
        let mut sum_squared_errors = 0.0;
        for i in 0..period {
            let x = i as f64;
            let y = self.bars[start_index + i].close;
            let predicted = intercept + slope * x;
            let error = y - predicted;
            sum_squared_errors += error * error;
        }
        
        let std_dev = (sum_squared_errors / (n - 2.0)).sqrt();
        
        Some((intercept, slope, std_dev))
    }
    
    fn calculate_all_t3_values(&mut self) {
        const T3_LENGTH: usize = 20;  // Changed from 8 to 20
        const SIGNAL_LENGTH: usize = 9;
        
        self.t3_values.clear();
        self.t3_fast_values.clear();
        self.t3_slow_values.clear();
        self.macd_values.clear();
        self.macd_signal.clear();
        self.macd_histogram.clear();
        
        if self.bars.is_empty() {
            return;
        }
        
        // Calculate volatility for all bars
        self.volatility_20.clear();
        for i in 0..self.bars.len() {
            if let Some(vol) = self.calculate_volatility(i, 20) {
                self.volatility_20.push(vol);
            } else {
                // Use previous value or default for early bars
                self.volatility_20.push(if i > 0 { self.volatility_20[i-1] } else { 3.0 });
            }
        }
        
        // Calculate T3 values for all bars
        for i in 0..self.bars.len() {
            // Standard T3 (20)
            if i < T3_LENGTH - 1 {
                self.t3_values.push(self.bars[i].close);
            } else {
                let t3 = self.calculate_t3(i);
                self.t3_values.push(t3);
            }
            
            // Fast T3
            if i < self.t3_fast_length - 1 {
                self.t3_fast_values.push(self.bars[i].close);
            } else {
                let t3_fast = self.calculate_t3_with_period(i, self.t3_fast_length);
                self.t3_fast_values.push(t3_fast);
            }
            
            // Slow T3
            if i < self.t3_slow_length - 1 {
                self.t3_slow_values.push(self.bars[i].close);
            } else {
                let t3_slow = self.calculate_t3_with_period(i, self.t3_slow_length);
                self.t3_slow_values.push(t3_slow);
            }
            
            // MACD = Fast - Slow
            if i >= self.t3_slow_length - 1 {
                let macd = self.t3_fast_values[i] - self.t3_slow_values[i];
                self.macd_values.push(macd);
                
                // Signal line (9-EMA of MACD)
                if self.macd_values.len() < SIGNAL_LENGTH {
                    self.macd_signal.push(macd);
                } else {
                    let alpha = 2.0 / (SIGNAL_LENGTH as f64 + 1.0);
                    let signal = alpha * macd + (1.0 - alpha) * self.macd_signal.last().unwrap();
                    self.macd_signal.push(signal);
                }
                
                // Histogram = MACD - Signal
                let histogram = macd - self.macd_signal.last().unwrap();
                self.macd_histogram.push(histogram);
            }
        }
    }
    
    fn calculate_t3(&self, bar_index: usize) -> f64 {
        self.calculate_t3_with_period(bar_index, 20)
    }
    
    fn recalculate_indicators(&mut self) {
        // Only recalculate up to current bar to improve performance
        self.calculate_t3_values_up_to(self.current_bar_index);
    }
    
    fn calculate_t3_values_up_to(&mut self, max_index: usize) {
        const SIGNAL_LENGTH: usize = 9;
        
        // Clear existing values
        self.t3_values.clear();
        self.t3_fast_values.clear();
        self.t3_slow_values.clear();
        self.macd_values.clear();
        self.macd_signal.clear();
        self.macd_histogram.clear();
        
        if self.bars.is_empty() {
            return;
        }
        
        let limit = max_index.min(self.bars.len() - 1) + 1;
        
        // Calculate T3 values only up to the current position
        for i in 0..limit {
            // Standard T3 (20)
            if i < 19 {
                self.t3_values.push(self.bars[i].close);
            } else {
                let t3 = self.calculate_t3(i);
                self.t3_values.push(t3);
            }
            
            // Fast T3
            if i < self.t3_fast_length - 1 {
                self.t3_fast_values.push(self.bars[i].close);
            } else {
                let t3_fast = self.calculate_t3_with_period(i, self.t3_fast_length);
                self.t3_fast_values.push(t3_fast);
            }
            
            // Slow T3
            if i < self.t3_slow_length - 1 {
                self.t3_slow_values.push(self.bars[i].close);
            } else {
                let t3_slow = self.calculate_t3_with_period(i, self.t3_slow_length);
                self.t3_slow_values.push(t3_slow);
            }
            
            // MACD = Fast - Slow
            if i >= self.t3_slow_length - 1 {
                let macd = self.t3_fast_values[i] - self.t3_slow_values[i];
                self.macd_values.push(macd);
                
                // Calculate signal line (9-EMA of MACD)
                if self.macd_values.len() < SIGNAL_LENGTH {
                    self.macd_signal.push(macd); // Use MACD value until we have enough data
                    self.macd_histogram.push(0.0);
                } else {
                    let alpha = 2.0 / (SIGNAL_LENGTH as f64 + 1.0);
                    let signal = if self.macd_signal.is_empty() {
                        macd
                    } else {
                        alpha * macd + (1.0 - alpha) * self.macd_signal.last().unwrap()
                    };
                    self.macd_signal.push(signal);
                    self.macd_histogram.push(macd - signal);
                }
            }
        }
    }
    
    fn calculate_t3_with_period(&self, bar_index: usize, period: usize) -> f64 {
        const T3_FACTOR: f64 = 0.7;
        
        if bar_index < period - 1 {
            return self.bars[bar_index].close;
        }
        
        // Calculate EMAs
        let mut ema1 = vec![0.0; bar_index + 1];
        let mut ema2 = vec![0.0; bar_index + 1];
        let mut ema3 = vec![0.0; bar_index + 1];
        let mut ema4 = vec![0.0; bar_index + 1];
        let mut ema5 = vec![0.0; bar_index + 1];
        let mut ema6 = vec![0.0; bar_index + 1];
        
        let alpha = 2.0 / (period as f64 + 1.0);
        
        // First EMA
        ema1[period - 1] = self.bars[..period].iter().map(|b| b.close).sum::<f64>() / period as f64;
        for i in period..=bar_index {
            ema1[i] = alpha * self.bars[i].close + (1.0 - alpha) * ema1[i - 1];
        }
        
        // Second EMA (EMA of first EMA)
        ema2[period - 1] = ema1[period - 1];
        for i in period..=bar_index {
            ema2[i] = alpha * ema1[i] + (1.0 - alpha) * ema2[i - 1];
        }
        
        // Continue for remaining EMAs
        ema3[period - 1] = ema2[period - 1];
        for i in period..=bar_index {
            ema3[i] = alpha * ema2[i] + (1.0 - alpha) * ema3[i - 1];
        }
        
        ema4[period - 1] = ema3[period - 1];
        for i in period..=bar_index {
            ema4[i] = alpha * ema3[i] + (1.0 - alpha) * ema4[i - 1];
        }
        
        ema5[period - 1] = ema4[period - 1];
        for i in period..=bar_index {
            ema5[i] = alpha * ema4[i] + (1.0 - alpha) * ema5[i - 1];
        }
        
        ema6[period - 1] = ema5[period - 1];
        for i in period..=bar_index {
            ema6[i] = alpha * ema5[i] + (1.0 - alpha) * ema6[i - 1];
        }
        
        // Calculate T3 using Tillson's formula
        let c1 = -T3_FACTOR * T3_FACTOR * T3_FACTOR;
        let c2 = 3.0 * T3_FACTOR * T3_FACTOR + 3.0 * T3_FACTOR * T3_FACTOR * T3_FACTOR;
        let c3 = -6.0 * T3_FACTOR * T3_FACTOR - 3.0 * T3_FACTOR - 3.0 * T3_FACTOR * T3_FACTOR * T3_FACTOR;
        let c4 = 1.0 + 3.0 * T3_FACTOR + T3_FACTOR * T3_FACTOR * T3_FACTOR + 3.0 * T3_FACTOR * T3_FACTOR;
        
        c1 * ema6[bar_index] + c2 * ema5[bar_index] + c3 * ema4[bar_index] + c4 * ema3[bar_index]
    }

    fn update_lot_size(&mut self) {
        if let Ok(size) = self.lot_size_string.parse::<i32>() {
            if size > 0 && size <= 10000 {
                self.lot_size = size;
            }
        }
    }

    fn calculate_unrealized_pnl(&self) -> f64 {
        if self.position == 0 || self.current_bar_index >= self.bars.len() {
            return 0.0;
        }
        
        let current_price = self.bars[self.current_bar_index].close;
        let (_, net_pnl) = self.calculate_pnl(current_price, self.position.abs(), self.position > 0);
        net_pnl
    }
    
    fn calculate_equity_tied_up(&self) -> f64 {
        if self.position == 0 || self.average_entry_price == 0.0 {
            return 0.0;
        }
        
        // For forex/stocks: position size * price
        // For crypto: same calculation
        self.position.abs() as f64 * self.average_entry_price
    }
    
    fn update_drawdown_tracking(&mut self) {
        if self.position != 0 {
            let unrealized = self.calculate_unrealized_pnl();
            
            // Track high water mark (best P&L seen)
            if unrealized > self.position_high_water_mark {
                self.position_high_water_mark = unrealized;
            }
            
            // Track max drawdown from high water mark
            let current_drawdown = self.position_high_water_mark - unrealized;
            if current_drawdown > self.max_unrealized_drawdown {
                self.max_unrealized_drawdown = current_drawdown;
            }
        }
    }

    fn calculate_pnl(&self, exit_price: f64, lots: i32, is_buy: bool) -> (f64, f64) {
        if self.average_entry_price == 0.0 {
            return (0.0, 0.0);
        }
        
        let price_diff = if is_buy {
            exit_price - self.average_entry_price
        } else {
            self.average_entry_price - exit_price
        };
        
        // Calculate gross P&L based on instrument type
        let gross_pnl = if self.instrument.contains("USD") && !self.instrument.starts_with("USD") {
            // Forex pairs like EUR/USD, GBP/USD
            price_diff * lots as f64 * 10000.0 // pip value
        } else if self.instrument.contains("BTC") || self.instrument.contains("ETH") {
            // Crypto - P&L is simply price difference * lots
            price_diff * lots as f64
        } else {
            // Default for stocks and other instruments
            price_diff * lots as f64
        };
        
        // Deduct slippage/spread costs based on instrument type
        let slippage_cost = if self.instrument.contains("USD") && !self.instrument.starts_with("USD") {
            // Forex pairs: slippage is in pips, convert to P&L
            self.slippage_per_lot * lots as f64 * 10000.0 // pip value in P&L terms
        } else {
            // Stocks/crypto: slippage is in price units per share
            self.slippage_per_lot * lots as f64 // direct cost per share
        };
        
        let net_pnl = gross_pnl - slippage_cost;
        (gross_pnl, net_pnl)
    }

    fn update_average_entry_price(&mut self, price: f64, lots: i32, is_buy: bool) {
        let _signed_lots = if is_buy { lots } else { -lots };
        
        if self.position == 0 {
            self.average_entry_price = price;
            self.average_price_history.push((self.current_bar_index, price));
            // Reset drawdown tracking for new position
            self.max_unrealized_drawdown = 0.0;
            self.position_high_water_mark = 0.0;
            // Track entry bar for closed trade visualization
            self.position_entry_bar = self.current_bar_index;
        } else if (self.position > 0 && is_buy) || (self.position < 0 && !is_buy) {
            // Adding to position
            let total_lots = self.position.abs() + lots;
            self.average_entry_price = (self.average_entry_price * self.position.abs() as f64 + price * lots as f64) / total_lots as f64;
            self.average_price_history.push((self.current_bar_index, self.average_entry_price));
        } else {
            // Reducing position - P&L is realized here
            let closed_lots = lots.min(self.position.abs());
            let (_, net_pnl) = self.calculate_pnl(price, closed_lots, self.position > 0);
            self.cumulative_pnl += net_pnl;
            
            // If position is reversed, set new average entry
            if lots > self.position.abs() {
                // Create closed trade record for the old position
                let was_long = self.position > 0;
                
                // Collect price history for this trade
                let mut trade_price_history = Vec::new();
                for &(bar_idx, avg_price) in &self.average_price_history {
                    if bar_idx >= self.position_entry_bar && avg_price > 0.0 {
                        trade_price_history.push((bar_idx, avg_price));
                    }
                }
                
                self.closed_trades.push(ClosedTrade {
                    entry_bar_index: self.position_entry_bar,
                    exit_bar_index: self.current_bar_index,
                    entry_price: self.average_entry_price,
                    exit_price: price,
                    pnl: net_pnl,
                    was_long,
                    price_history: trade_price_history,
                });
                
                // Mark the end of the previous position
                if self.average_entry_price > 0.0 {
                    self.average_price_history.push((self.current_bar_index, 0.0));
                }
                self.average_entry_price = price;
                self.average_price_history.push((self.current_bar_index, price));
                self.position_entry_bar = self.current_bar_index;
                // Reset drawdown tracking for new position
                self.max_unrealized_drawdown = 0.0;
                self.position_high_water_mark = 0.0;
            } else if self.position.abs() - lots == 0 {
                // Position fully closed
                let was_long = self.position > 0;
                
                // Collect price history for this trade
                let mut trade_price_history = Vec::new();
                for &(bar_idx, avg_price) in &self.average_price_history {
                    if bar_idx >= self.position_entry_bar && avg_price > 0.0 {
                        trade_price_history.push((bar_idx, avg_price));
                    }
                }
                
                self.closed_trades.push(ClosedTrade {
                    entry_bar_index: self.position_entry_bar,
                    exit_bar_index: self.current_bar_index,
                    entry_price: self.average_entry_price,
                    exit_price: price,
                    pnl: net_pnl,
                    was_long,
                    price_history: trade_price_history,
                });
                
                if self.average_entry_price > 0.0 {
                    self.average_price_history.push((self.current_bar_index, 0.0));
                }
                self.average_entry_price = 0.0;
                // Reset drawdown tracking
                self.max_unrealized_drawdown = 0.0;
                self.position_high_water_mark = 0.0;
            }
        }
    }

    fn buy(&mut self) {
        // Check if trading is allowed based on market regime
        if !self.is_trading_allowed() {
            return;
        }
        
        // If delay entry is enabled, set pending order instead of executing immediately
        if self.delay_entry_consecutive_bars {
            self.pending_buy_order = true;
            self.pending_sell_order = false;  // Cancel any pending sell
            return;
        }
        
        // Check if we should disable trading when positioned
        if self.disable_when_positioned && self.position != 0 {
            return;
        }
        
        // Check if we can open a trade at this time
        if self.auto_flatten_at_close && self.current_bar_index < self.bars.len() {
            let current_bar = &self.bars[self.current_bar_index];
            if !can_open_trade(&current_bar.timestamp, &self.bars, self.current_bar_index) {
                return;
            }
        }
        
        if self.current_bar_index < self.bars.len() {
            let timestamp = self.bars[self.current_bar_index].timestamp;
            let close_price = self.bars[self.current_bar_index].close;
            let old_position = self.position;
            
            // Calculate P&L if closing a short position
            let (gross_pnl, trade_pnl) = if old_position < 0 {
                let closed_lots = self.lot_size.min(old_position.abs());
                self.calculate_pnl(close_price, closed_lots, false)
            } else {
                (0.0, 0.0)
            };
            
            self.update_average_entry_price(close_price, self.lot_size, true);
            self.position += self.lot_size;
            
            let transaction = Transaction {
                timestamp,
                instrument: self.instrument.clone(),
                action: TradeAction::Buy,
                lots: self.lot_size,
                price: close_price,
                position_after: self.position,
                trade_pnl,
                gross_pnl,
                cumulative_pnl: self.cumulative_pnl,
                max_unrealized_drawdown: self.max_unrealized_drawdown,
                equity_tied_up: self.calculate_equity_tied_up(),
            };
            self.log_transaction(&transaction);
            self.transactions.push(transaction);
        }
    }

    fn sell(&mut self) {
        // Check if trading is allowed based on market regime
        if !self.is_trading_allowed() {
            return;
        }
        
        // If delay entry is enabled, set pending order instead of executing immediately
        if self.delay_entry_consecutive_bars {
            self.pending_sell_order = true;
            self.pending_buy_order = false;  // Cancel any pending buy
            return;
        }
        
        // Check if we should disable trading when positioned
        if self.disable_when_positioned && self.position != 0 {
            return;
        }
        
        // Check if we can open a trade at this time
        if self.auto_flatten_at_close && self.current_bar_index < self.bars.len() {
            let current_bar = &self.bars[self.current_bar_index];
            if !can_open_trade(&current_bar.timestamp, &self.bars, self.current_bar_index) {
                return;
            }
        }
        
        if self.current_bar_index < self.bars.len() {
            let timestamp = self.bars[self.current_bar_index].timestamp;
            let close_price = self.bars[self.current_bar_index].close;
            let old_position = self.position;
            
            // Calculate P&L if closing a long position
            let (gross_pnl, trade_pnl) = if old_position > 0 {
                let closed_lots = self.lot_size.min(old_position.abs());
                self.calculate_pnl(close_price, closed_lots, true)
            } else {
                (0.0, 0.0)
            };
            
            self.update_average_entry_price(close_price, self.lot_size, false);
            self.position -= self.lot_size;
            
            let transaction = Transaction {
                timestamp,
                instrument: self.instrument.clone(),
                action: TradeAction::Sell,
                lots: self.lot_size,
                price: close_price,
                position_after: self.position,
                trade_pnl,
                gross_pnl,
                cumulative_pnl: self.cumulative_pnl,
                max_unrealized_drawdown: self.max_unrealized_drawdown,
                equity_tied_up: self.calculate_equity_tied_up(),
            };
            self.log_transaction(&transaction);
            self.transactions.push(transaction);
        }
    }

    fn flatten(&mut self) {
        // Cancel any pending orders when flattening
        self.pending_buy_order = false;
        self.pending_sell_order = false;
        
        if self.current_bar_index < self.bars.len() && self.position != 0 {
            let timestamp = self.bars[self.current_bar_index].timestamp;
            let close_price = self.bars[self.current_bar_index].close;
            let lots = self.position.abs();
            
            // Calculate P&L for closing entire position
            let (gross_pnl, trade_pnl) = self.calculate_pnl(close_price, lots, self.position > 0);
            self.cumulative_pnl += trade_pnl;
            
            // Create closed trade record for visualization
            let was_long = self.position > 0;
            
            // Collect price history for this trade
            let mut trade_price_history = Vec::new();
            for &(bar_idx, price) in &self.average_price_history {
                if bar_idx >= self.position_entry_bar && price > 0.0 {
                    trade_price_history.push((bar_idx, price));
                }
            }
            
            self.closed_trades.push(ClosedTrade {
                entry_bar_index: self.position_entry_bar,
                exit_bar_index: self.current_bar_index,
                entry_price: self.average_entry_price,
                exit_price: close_price,
                pnl: trade_pnl,
                was_long,
                price_history: trade_price_history,
            });
            
            self.position = 0;
            // Add a final entry to mark where the position was closed
            if self.average_entry_price > 0.0 {
                self.average_price_history.push((self.current_bar_index, 0.0)); // 0.0 marks the end
            }
            self.average_entry_price = 0.0;
            // Don't clear average_price_history - keep it for display until reset
            
            let transaction = Transaction {
                timestamp,
                instrument: self.instrument.clone(),
                action: TradeAction::Flatten,
                lots,
                price: close_price,
                position_after: 0,
                trade_pnl,
                gross_pnl,
                cumulative_pnl: self.cumulative_pnl,
                max_unrealized_drawdown: self.max_unrealized_drawdown, // Keep the drawdown from the position
                equity_tied_up: 0.0, // No equity tied up after flattening
            };
            self.log_transaction(&transaction);
            self.transactions.push(transaction);
            
            // Reset drawdown tracking after flattening
            self.max_unrealized_drawdown = 0.0;
            self.position_high_water_mark = 0.0;
        }
    }

    fn log_transaction(&self, transaction: &Transaction) {
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open("transactions.json")
        {
            let json = serde_json::to_string(transaction).unwrap();
            writeln!(file, "{}", json).ok();
        }
    }
    
    fn copy_transactions_as_csv(&self, ui: &egui::Ui) {
        let mut csv_content = String::new();
        
        // Add header
        csv_content.push_str("Timestamp,Instrument,Action,Lots,Price,Position,Gross P&L,Net P&L,Total P&L,Max Drawdown,Tied Up\n");
        
        // Add each transaction
        for transaction in &self.transactions {
            csv_content.push_str(&format!(
                "{},{},{:?},{},{:.5},{},{:.2},{:.2},{:.2},{:.2},{:.2}\n",
                transaction.timestamp.with_timezone(&self.display_timezone).format("%Y-%m-%d %H:%M:%S"),
                transaction.instrument,
                transaction.action,
                transaction.lots,
                transaction.price,
                transaction.position_after,
                transaction.gross_pnl,
                transaction.trade_pnl,
                transaction.cumulative_pnl,
                transaction.max_unrealized_drawdown,
                transaction.equity_tied_up
            ));
        }
        
        // Copy to clipboard
        ui.output_mut(|o| o.copied_text = csv_content);
    }
}

impl eframe::App for TradingReplayApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Check if any text input wants keyboard input
        let text_input_wants_keyboard = ctx.wants_keyboard_input();
        
        // Handle hotkeys - Arrow keys for trading (only if no text input is focused)
        if !text_input_wants_keyboard {
            if ctx.input(|i| i.key_pressed(egui::Key::ArrowRight)) {
                self.buy();
            }
            if ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft)) {
                self.sell();
            }
            if ctx.input(|i| i.key_pressed(egui::Key::ArrowDown)) {
                self.flatten();
            }
        }
        
        // Handle zoom controls
        ctx.input(|i| {
            let ctrl_held = i.modifiers.ctrl || i.modifiers.command;
            
            // Ctrl+Plus/Minus for zoom
            if ctrl_held {
                if i.key_pressed(egui::Key::Plus) || i.key_pressed(egui::Key::Equals) {
                    self.zoom_level = (self.zoom_level * 1.2).min(5.0);
                }
                if i.key_pressed(egui::Key::Minus) {
                    self.zoom_level = (self.zoom_level / 1.2).max(0.05);
                }
                
                // Ctrl+MouseWheel for zoom
                let scroll_delta = i.raw_scroll_delta.y;
                if scroll_delta != 0.0 {
                    if scroll_delta > 0.0 {
                        self.zoom_level = (self.zoom_level * 1.1).min(5.0);
                    } else {
                        self.zoom_level = (self.zoom_level / 1.1).max(0.05);
                    }
                }
            }
        });

        // Handle scrolling when paused or at the end
        if !self.playing || self.current_bar_index >= self.bars.len() - 1 {
            // Page Up/Down for scrolling since arrow keys are now for trading
            if ctx.input(|i| i.key_pressed(egui::Key::PageUp)) {
                self.view_offset = (self.view_offset - 5).max(-(self.current_bar_index as i32));
            }
            if ctx.input(|i| i.key_pressed(egui::Key::PageDown)) {
                self.view_offset = (self.view_offset + 5).min(10); // Can scroll up to 10 bars into the future
            }
            
            // Mouse wheel scrolling will be handled in the chart area only
        } else {
            // When playing, default view is 10 bars ahead but allow temporary scrolling
            self.view_offset = 10;
        }

        // Update replay
        if self.playing && self.last_update.elapsed().as_secs_f32() > (1.0 / self.speed) {
            if self.current_bar_index < self.bars.len() - 1 {
                self.current_bar_index += 1;
                self.last_update = std::time::Instant::now();
                
                // FAILSAFE: If auto-flatten is enabled and we have a position, 
                // check if we're outside market hours and flatten immediately
                if self.auto_flatten_at_close && self.position != 0 && self.current_bar_index < self.bars.len() {
                    let current_bar = &self.bars[self.current_bar_index];
                    if !is_stock_market_hours(&current_bar.timestamp) {
                        self.flatten();
                    }
                }
                
                // Update drawdown tracking if we have a position
                self.update_drawdown_tracking();
                
                // Check for auto-flatten on profit
                if self.auto_flatten_on_profit && self.position != 0 {
                    let unrealized = self.calculate_unrealized_pnl();
                    
                    // Calculate minimum profit threshold based on position size and slippage
                    let min_profit = if self.instrument.contains("USD") && !self.instrument.starts_with("USD") {
                        // Forex: slippage in pips, convert to P&L
                        self.slippage_per_lot * self.position.abs() as f64 * 10000.0
                    } else {
                        // Stocks/crypto: slippage in price units
                        self.slippage_per_lot * self.position.abs() as f64
                    };
                    
                    if unrealized > min_profit {
                        self.flatten();
                    }
                }
                
                // Check for stop loss and take profit
                if self.position != 0 {
                    let unrealized = self.calculate_unrealized_pnl();
                    
                    // Check stop loss
                    if let Some(stop_loss) = self.stop_loss_dollars {
                        if unrealized <= -stop_loss {
                            eprintln!("STOP LOSS: Flattening position at ${:.2} loss", -unrealized);
                            self.flatten();
                        }
                    }
                    
                    // Check take profit (only if we didn't just flatten from stop loss)
                    if self.position != 0 {
                        if let Some(take_profit) = self.take_profit_dollars {
                            if unrealized >= take_profit {
                                eprintln!("TAKE PROFIT: Flattening position at ${:.2} profit", unrealized);
                                self.flatten();
                            }
                        }
                    }
                }
                
                // Check for auto-flatten at market close
                if self.auto_flatten_at_close && self.position != 0 {
                    let current_bar = &self.bars[self.current_bar_index];
                    let next_bar = self.bars.get(self.current_bar_index + 1).map(|b| &b.timestamp);
                    
                    // Debug: Show when we're checking
                    if is_market_close(&current_bar.timestamp, next_bar) {
                        eprintln!("AUTO-FLATTEN: Market close detected at {}", 
                                 current_bar.timestamp.with_timezone(&self.display_timezone).format("%H:%M"));
                        self.flatten();
                    }
                }
                
                // Delayed Entry - Check for 3 consecutive bars pattern
                if self.delay_entry_consecutive_bars && self.current_bar_index >= 3 {
                    // Check last 3 bars
                    let bar1 = &self.bars[self.current_bar_index - 2];
                    let bar2 = &self.bars[self.current_bar_index - 1];
                    let bar3 = &self.bars[self.current_bar_index];
                    
                    let bar1_green = bar1.close > bar1.open;
                    let bar2_green = bar2.close > bar2.open;
                    let bar3_green = bar3.close > bar3.open;
                    
                    // Execute pending buy after 3 green bars
                    if self.pending_buy_order && bar1_green && bar2_green && bar3_green {
                        // Turn off delay temporarily to execute the order
                        self.delay_entry_consecutive_bars = false;
                        self.buy();
                        self.delay_entry_consecutive_bars = true;
                        self.pending_buy_order = false;
                    }
                    // Execute pending sell after 3 red bars
                    else if self.pending_sell_order && !bar1_green && !bar2_green && !bar3_green {
                        // Turn off delay temporarily to execute the order
                        self.delay_entry_consecutive_bars = false;
                        self.sell();
                        self.delay_entry_consecutive_bars = true;
                        self.pending_sell_order = false;
                    }
                }
                
                // Update market regime
                self.update_market_regime();
                
                // Execute strategy signals if enabled
                if self.enable_strategy_signals {
                    if let Some(signal) = self.get_strategy_signal() {
                        self.last_strategy_signal = Some(signal);
                        if signal {
                            // Buy signal
                            if self.position <= 0 {
                                if self.position < 0 {
                                    self.flatten();
                                }
                                self.buy();
                            }
                        } else {
                            // Sell signal
                            if self.position >= 0 {
                                if self.position > 0 {
                                    self.flatten();
                                }
                                self.sell();
                            }
                        }
                    }
                }
                
                // Update T3 and MACD values
                if self.current_bar_index >= self.t3_values.len() {
                    let t3 = self.calculate_t3(self.current_bar_index);
                    self.t3_values.push(t3);
                    
                    // Update fast T3
                    let t3_fast = self.calculate_t3_with_period(self.current_bar_index, self.t3_fast_length);
                    self.t3_fast_values.push(t3_fast);
                    
                    // Update slow T3
                    let t3_slow = self.calculate_t3_with_period(self.current_bar_index, self.t3_slow_length);
                    self.t3_slow_values.push(t3_slow);
                    
                    // Update MACD
                    if self.current_bar_index >= self.t3_slow_length - 1 { // Need enough bars for slow T3
                        let macd = t3_fast - t3_slow;
                        self.macd_values.push(macd);
                        
                        // Update signal line
                        const SIGNAL_LENGTH: usize = 9;
                        if self.macd_values.len() < SIGNAL_LENGTH {
                            self.macd_signal.push(macd);
                        } else {
                            let alpha = 2.0 / (SIGNAL_LENGTH as f64 + 1.0);
                            let signal = alpha * macd + (1.0 - alpha) * self.macd_signal.last().unwrap();
                            self.macd_signal.push(signal);
                        }
                        
                        // Update histogram
                        let histogram = macd - self.macd_signal.last().unwrap();
                        self.macd_histogram.push(histogram);
                    }
                }
            } else {
                self.playing = false;
            }
        }

        // Show data loading menu if requested
        if self.show_data_menu {
            egui::Window::new("Load Market Data")
                .collapsible(false)
                .resizable(true)
                .show(ctx, |ui| {
                    ui.heading("Data Sources");
                    
                    ui.separator();
                    
                    if ui.button("📂 Load from CSV/JSON file").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Data files", &["csv", "json"])
                            .add_filter("CSV files", &["csv"])
                            .add_filter("JSON files", &["json"])
                            .pick_file()
                        {
                            self.load_data_from_file(path);
                        }
                    }
                    
                    ui.separator();
                    
                    ui.label("Recently loaded:");
                    if let Some(path) = &self.data_path {
                        ui.label(format!("📄 {}", path.display()));
                    } else {
                        ui.label("None");
                    }
                    
                    ui.separator();
                    
                    if let Some(error) = &self.load_error {
                        ui.colored_label(egui::Color32::RED, format!("Error: {}", error));
                    }
                    
                    ui.separator();
                    
                    ui.collapsing("📖 Instructions", |ui| {
                        ui.label(data_loader::create_tradingview_export_instructions());
                    });
                    
                    ui.separator();
                    
                    ui.label(format!("Current data: {} bars", self.bars.len()));
                    if !self.bars.is_empty() {
                        let first = &self.bars[0].timestamp;
                        let last = &self.bars[self.bars.len() - 1].timestamp;
                        ui.label(format!("Date range: {} to {}", 
                            first.format("%Y-%m-%d"),
                            last.format("%Y-%m-%d")
                        ));
                    }
                    
                    ui.separator();
                    
                    if ui.button("Close").clicked() {
                        self.show_data_menu = false;
                    }
                });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            // Header with instrument and OHLC info
            ui.horizontal_wrapped(|ui| {
                ui.heading(&self.instrument);
                ui.separator();
                if self.current_bar_index < self.bars.len() {
                    let current_bar = &self.bars[self.current_bar_index];
                    let local_time = current_bar.timestamp.with_timezone(&self.display_timezone);
                    ui.label(format!(
                        "{} | O: {:.5} H: {:.5} L: {:.5} C: {:.5}",
                        local_time.format("%H:%M"),
                        current_bar.open,
                        current_bar.high,
                        current_bar.low,
                        current_bar.close
                    ));
                    
                    // Add MACD info
                    if self.current_bar_index >= self.t3_slow_length - 1 {
                        let macd_idx = self.current_bar_index - (self.t3_slow_length - 1);
                        if macd_idx < self.macd_values.len() {
                            let macd = self.macd_values[macd_idx];
                            let signal = self.macd_signal.get(macd_idx).copied().unwrap_or(macd);
                            let histogram = self.macd_histogram.get(macd_idx).copied().unwrap_or(0.0);
                            
                            ui.separator();
                            ui.label(format!(
                                "MACD: {:.5} Signal: {:.5} Hist: {:.5}",
                                macd, signal, histogram
                            ));
                        }
                    }
                }
            });

            ui.separator();

            // Controls
            egui::ScrollArea::horizontal()
                .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
                .show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("📁 Load Data").clicked() {
                    self.show_data_menu = !self.show_data_menu;
                }
                ui.separator();
                if ui.button(if self.playing { "⏸ Pause" } else { "▶ Play" }).clicked() {
                    self.playing = !self.playing;
                }
                ui.separator();
                ui.label("Speed:");
                ui.add(egui::Slider::new(&mut self.speed, 0.1..=20.0).suffix("x"));
                ui.separator();
                ui.label("Lot Size:");
                let lot_response = ui.add_sized(
                    egui::vec2(60.0, 20.0),
                    egui::TextEdit::singleline(&mut self.lot_size_string)
                );
                if lot_response.changed() || lot_response.lost_focus() {
                    self.update_lot_size();
                }
                ui.separator();
                ui.label(format!("Position: {}", self.position));
                
                // Show pending orders
                if self.pending_buy_order || self.pending_sell_order {
                    ui.separator();
                    let pending_text = if self.pending_buy_order {
                        "Pending BUY (waiting for 3 green bars)"
                    } else {
                        "Pending SELL (waiting for 3 red bars)"
                    };
                    ui.colored_label(egui::Color32::YELLOW, pending_text);
                }
                
                ui.separator();
                
                // Show unrealized P&L if position is open
                if self.position != 0 {
                    let unrealized = self.calculate_unrealized_pnl();
                    let color = if unrealized >= 0.0 { 
                        egui::Color32::GREEN 
                    } else { 
                        egui::Color32::RED 
                    };
                    ui.colored_label(color, format!("Unrealized: ${:.2}", unrealized));
                    ui.separator();
                    
                    // Show stop loss and take profit info
                    if let Some(stop_loss) = self.stop_loss_dollars {
                        let distance_to_stop = unrealized + stop_loss;
                        let stop_color = if distance_to_stop < stop_loss * 0.2 { 
                            egui::Color32::YELLOW // Warning when close to stop
                        } else { 
                            egui::Color32::from_gray(150) 
                        };
                        ui.colored_label(stop_color, format!("SL: -${:.0} (${:.2} away)", stop_loss, distance_to_stop));
                    }
                    
                    if let Some(take_profit) = self.take_profit_dollars {
                        let distance_to_tp = take_profit - unrealized;
                        let tp_color = if distance_to_tp < take_profit * 0.2 { 
                            egui::Color32::LIGHT_GREEN // Highlight when close to TP
                        } else { 
                            egui::Color32::from_gray(150) 
                        };
                        ui.colored_label(tp_color, format!("TP: +${:.0} (${:.2} away)", take_profit, distance_to_tp));
                    }
                    ui.separator();
                }
                
                ui.label(format!("Cumulative P&L: ${:.2}", self.cumulative_pnl));
                ui.separator();
                ui.label(format!("Zoom: {:.0}%", self.zoom_level * 100.0));
                ui.separator();
                
                // Date jump input
                ui.label("Jump to:");
                let response = ui.add_sized(
                    egui::vec2(100.0, 20.0),
                    egui::TextEdit::singleline(&mut self.start_date_input)
                        .hint_text("YYYY-MM-DD")
                );
                if ui.button("Go").clicked() || (response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))) {
                    self.jump_to_date();
                }
                ui.separator();
                
                // Slippage/spread setting (applies to all trades)
                ui.horizontal(|ui| {
                    let label = if self.instrument.contains("USD") && !self.instrument.starts_with("USD") {
                        "Slippage (pips):"
                    } else {
                        "Slippage ($):"
                    };
                    ui.label(label);
                    let mut slippage_string = format!("{:.5}", self.slippage_per_lot);
                    let response = ui.add_sized(
                        egui::vec2(80.0, 20.0),
                        egui::TextEdit::singleline(&mut slippage_string)
                    );
                    if response.changed() || response.lost_focus() {
                        if let Ok(value) = slippage_string.parse::<f64>() {
                            self.slippage_per_lot = value.max(0.0);
                        }
                    }
                    
                    // Show example calculation
                    if self.position != 0 {
                        let example_cost = if self.instrument.contains("USD") && !self.instrument.starts_with("USD") {
                            self.slippage_per_lot * 10000.0 // pip to dollar conversion
                        } else {
                            self.slippage_per_lot // direct dollar cost
                        };
                        ui.label(format!("(${:.2}/lot)", example_cost));
                    }
                });
                
                ui.separator();
                
                // T3 length configuration
                ui.horizontal(|ui| {
                    ui.label("T3 Fast:");
                    let response = ui.add_sized(
                        egui::vec2(40.0, 20.0),
                        egui::TextEdit::singleline(&mut self.t3_fast_length_string)
                    );
                    if response.lost_focus() || (response.changed() && ui.input(|i| i.key_pressed(egui::Key::Enter))) {
                        if let Ok(value) = self.t3_fast_length_string.parse::<usize>() {
                            if value >= 2 && value <= 50 && value != self.t3_fast_length {
                                self.t3_fast_length = value;
                                self.recalculate_indicators();
                            }
                        } else {
                            self.t3_fast_length_string = self.t3_fast_length.to_string();
                        }
                    }
                    
                    ui.separator();
                    ui.label("T3 Slow:");
                    let response = ui.add_sized(
                        egui::vec2(40.0, 20.0),
                        egui::TextEdit::singleline(&mut self.t3_slow_length_string)
                    );
                    if response.lost_focus() || (response.changed() && ui.input(|i| i.key_pressed(egui::Key::Enter))) {
                        if let Ok(value) = self.t3_slow_length_string.parse::<usize>() {
                            if value >= 2 && value <= 100 && value >= self.t3_fast_length && value != self.t3_slow_length {
                                self.t3_slow_length = value;
                                self.recalculate_indicators();
                            }
                        } else {
                            self.t3_slow_length_string = self.t3_slow_length.to_string();
                        }
                    }
                });
                
                // Auto-flatten checkbox
                ui.checkbox(&mut self.auto_flatten_on_profit, "Auto-flatten on profit");
                
                ui.separator();
                
                // Disable when positioned checkbox
                ui.checkbox(&mut self.disable_when_positioned, "Disable buy/sell when positioned");
                
                // Auto-flatten at market close checkbox
                ui.checkbox(&mut self.auto_flatten_at_close, "Auto-flatten at market close");
                
                ui.separator();
                
                // Indicator controls
                ui.checkbox(&mut self.show_macd, "MACD");
                ui.checkbox(&mut self.show_linreg, "Linear Regression");
                
                ui.separator();
                
                // Delayed entry
                ui.checkbox(&mut self.delay_entry_consecutive_bars, "Delay Entry (3 Bars)");
                
                ui.separator();
                
                // Market Regime
                ui.checkbox(&mut self.show_regime_indicator, "Show Regime");
                if self.show_regime_indicator {
                    ui.horizontal(|ui| {
                        let regime_text = match self.current_regime {
                            MarketRegime::LowVolatility => "Low Vol",
                            MarketRegime::HighVolatility => "HIGH VOL",
                        };
                        let regime_color = match self.current_regime {
                            MarketRegime::LowVolatility => egui::Color32::GREEN,
                            MarketRegime::HighVolatility => egui::Color32::RED,
                        };
                        ui.colored_label(regime_color, regime_text);
                        
                        // Show current volatility
                        if self.current_bar_index < self.volatility_20.len() {
                            let current_vol = self.volatility_20[self.current_bar_index];
                            ui.label(format!("({:.1}%)", current_vol));
                        }
                    });
                    
                    ui.horizontal(|ui| {
                        ui.label("Vol Threshold:");
                        ui.add(egui::Slider::new(&mut self.regime_threshold, 2.0..=10.0)
                            .suffix("%")
                            .step_by(0.5));
                    });
                    
                    ui.horizontal(|ui| {
                        ui.label("Low Vol Strategy:");
                        ui.selectable_value(&mut self.low_vol_strategy, TradingStrategy::Momentum, "Momentum");
                        ui.selectable_value(&mut self.low_vol_strategy, TradingStrategy::MeanRevert, "Mean Revert");
                        ui.selectable_value(&mut self.low_vol_strategy, TradingStrategy::NoTrade, "No Trade");
                    });
                    
                    ui.horizontal(|ui| {
                        ui.label("High Vol Strategy:");
                        ui.selectable_value(&mut self.high_vol_strategy, TradingStrategy::Momentum, "Momentum");
                        ui.selectable_value(&mut self.high_vol_strategy, TradingStrategy::MeanRevert, "Mean Revert");
                        ui.selectable_value(&mut self.high_vol_strategy, TradingStrategy::NoTrade, "No Trade");
                    });
                    
                    ui.checkbox(&mut self.enable_strategy_signals, "Auto Trade Strategies");
                    
                    // Show last signal if auto trading is enabled
                    if self.enable_strategy_signals {
                        if let Some(signal) = self.last_strategy_signal {
                            let signal_text = if signal { "Last: BUY signal" } else { "Last: SELL signal" };
                            let signal_color = if signal { egui::Color32::GREEN } else { egui::Color32::RED };
                            ui.colored_label(signal_color, signal_text);
                        }
                    }
                }
                
                ui.separator();
                
                // Stop Loss and Take Profit controls
                ui.horizontal(|ui| {
                    ui.label("Stop Loss $:");
                    let response = ui.add_sized(
                        egui::vec2(60.0, 20.0),
                        egui::TextEdit::singleline(&mut self.stop_loss_string)
                            .hint_text("e.g. 100")
                    );
                    if response.lost_focus() || (response.changed() && ui.input(|i| i.key_pressed(egui::Key::Enter))) {
                        if self.stop_loss_string.is_empty() {
                            self.stop_loss_dollars = None;
                        } else if let Ok(value) = self.stop_loss_string.parse::<f64>() {
                            if value > 0.0 {
                                self.stop_loss_dollars = Some(value);
                            }
                        } else {
                            self.stop_loss_string = self.stop_loss_dollars.map(|v| v.to_string()).unwrap_or_default();
                        }
                    }
                    
                    ui.separator();
                    ui.label("Take Profit $:");
                    let response = ui.add_sized(
                        egui::vec2(60.0, 20.0),
                        egui::TextEdit::singleline(&mut self.take_profit_string)
                            .hint_text("e.g. 200")
                    );
                    if response.lost_focus() || (response.changed() && ui.input(|i| i.key_pressed(egui::Key::Enter))) {
                        if self.take_profit_string.is_empty() {
                            self.take_profit_dollars = None;
                        } else if let Ok(value) = self.take_profit_string.parse::<f64>() {
                            if value > 0.0 {
                                self.take_profit_dollars = Some(value);
                            }
                        } else {
                            self.take_profit_string = self.take_profit_dollars.map(|v| v.to_string()).unwrap_or_default();
                        }
                    }
                });
                
            });
            });

            ui.separator();

            // Chart area
            let available_size = ui.available_size();
            let chart_height = if self.show_macd {
                (available_size.y * 0.6).max(100.0)  // Main chart height when MACD is shown
            } else {
                (available_size.y * 0.8).max(100.0)  // Larger chart when MACD is hidden
            };
            let macd_height = (available_size.y * 0.2).max(50.0);  // MACD panel height
            
            // Define constants at the top level
            const BASE_BAR_WIDTH: f32 = 10.0;
            
            // Create a group to contain the chart with proper margins
            ui.group(|ui| {
                // Use all available width with small margins
                let chart_width = available_size.x.max(100.0);  // Minimum width
                let chart_size = egui::Vec2::new(chart_width, chart_height);
                let (response, painter) = ui.allocate_painter(chart_size, egui::Sense::hover());
                let chart_rect = response.rect;
                
                // Draw regime background if enabled
                if self.show_regime_indicator && self.current_bar_index < self.volatility_20.len() {
                    let current_vol = self.volatility_20[self.current_bar_index];
                    let bg_color = if current_vol > self.regime_threshold {
                        egui::Color32::from_rgba_unmultiplied(255, 0, 0, 20) // Red tint for high vol
                    } else {
                        egui::Color32::from_rgba_unmultiplied(0, 255, 0, 10) // Green tint for low vol
                    };
                    painter.rect_filled(chart_rect, 0.0, bg_color);
                }
                
                // Handle scrolling only when mouse is over the chart
                if response.hovered() && !self.playing {
                    ctx.input(|i| {
                        if !i.modifiers.ctrl && !i.modifiers.command {
                            let scroll_delta = i.raw_scroll_delta.x + i.raw_scroll_delta.y; // Support both axes
                            if scroll_delta != 0.0 {
                                self.view_offset = (self.view_offset + (scroll_delta * 0.5) as i32)
                                    .max(-(self.current_bar_index as i32))
                                    .min(10); // Can scroll up to 10 bars into the future
                            }
                        }
                    });
                }
                
                if !self.bars.is_empty() {
                let bar_width = BASE_BAR_WIDTH * self.zoom_level;
                
                // Calculate how many bars can fit in the window at current zoom
                let bars_that_fit = (chart_rect.width() / bar_width).floor() as i32;
                
                // Calculate what bars to show based on current position and offset
                // When offset is positive, we're showing empty space to the right (like vim's virtual lines)
                // The rightmost_bar is the position we're viewing, which can be beyond actual data
                let rightmost_bar_position = self.current_bar_index as i32 + self.view_offset;
                let leftmost_bar_position = rightmost_bar_position - bars_that_fit + 1;
                
                // Get the actual bar data to display (limited by available data AND current position)
                // When we have a positive offset, we don't want to show bars beyond current_bar_index
                let start_idx = leftmost_bar_position.max(0) as usize;
                let end_idx = if self.view_offset > 0 {
                    // During replay or when scrolled ahead, don't show bars beyond current position
                    self.bars.len().min((self.current_bar_index + 1).min(rightmost_bar_position.max(0) as usize + 1))
                } else {
                    // When scrolled back, show all available bars
                    self.bars.len().min(rightmost_bar_position.max(0) as usize + 1)
                };
                
                // Always draw the chart area, even if no bars are visible
                {
                    let visible_data = &self.bars[start_idx..end_idx];
                    let _actual_visible_bars = visible_data.len();
                    
                    // Calculate price range - use current bar if no visible data
                    let (mut min_price, mut max_price) = if visible_data.is_empty() && self.current_bar_index < self.bars.len() {
                        let current_bar = &self.bars[self.current_bar_index];
                        (current_bar.low - 0.0010, current_bar.high + 0.0010)
                    } else if !visible_data.is_empty() {
                        visible_data.iter()
                            .fold((f64::MAX, f64::MIN), |(min, max), bar| {
                                (min.min(bar.low), max.max(bar.high))
                            })
                    } else {
                        (1.0, 2.0) // Default range if no data
                    };
                    
                    
                    // Consider average entry price in the price range
                    if self.position != 0 && self.average_entry_price > 0.0 {
                        min_price = min_price.min(self.average_entry_price);
                        max_price = max_price.max(self.average_entry_price);
                    }
                    
                    // Add padding to price range to prevent bars from touching edges
                    let price_padding = (max_price - min_price) * 0.1;
                    min_price -= price_padding;
                    max_price += price_padding;
                    
                    let price_range = max_price - min_price;
                    
                    // Draw closed trades first (as background)
                    for trade in &self.closed_trades {
                        // Check if this trade is visible in current view
                        let entry_pos = trade.entry_bar_index as i32 - leftmost_bar_position;
                        let exit_pos = trade.exit_bar_index as i32 - leftmost_bar_position;
                        
                        // Only draw if at least part of the trade is visible
                        if exit_pos >= 0 && entry_pos < bars_that_fit {
                            let entry_x = chart_rect.left() + entry_pos as f32 * bar_width;
                            let exit_x = chart_rect.left() + exit_pos as f32 * bar_width;
                            
                            let entry_y = chart_rect.top() + (1.0 - (trade.entry_price - min_price) / price_range) as f32 * chart_rect.height();
                            let exit_y = chart_rect.top() + (1.0 - (trade.exit_price - min_price) / price_range) as f32 * chart_rect.height();
                            
                            // Draw exit line
                            let line_color = if trade.was_long {
                                egui::Color32::from_rgba_unmultiplied(0, 200, 0, 150)
                            } else {
                                egui::Color32::from_rgba_unmultiplied(200, 0, 0, 150)
                            };
                            
                            // Exit line (constant across the trade)
                            painter.line_segment(
                                [egui::pos2(entry_x.max(chart_rect.left()), exit_y), 
                                 egui::pos2(exit_x.min(chart_rect.right()), exit_y)],
                                egui::Stroke::new(2.0, line_color),
                            );
                            
                            // Fill area between entry and exit, following average price changes
                            let fill_color = if trade.pnl >= 0.0 {
                                egui::Color32::from_rgba_unmultiplied(0, 255, 0, 10)
                            } else {
                                egui::Color32::from_rgba_unmultiplied(255, 0, 0, 10)
                            };
                            
                            // Use the stored price history for this trade
                            let price_segments = &trade.price_history;
                            
                            // Draw filled polygons for each segment
                            for i in 0..price_segments.len() {
                                let (start_bar, avg_price) = price_segments[i];
                                let end_bar = if i + 1 < price_segments.len() {
                                    price_segments[i + 1].0
                                } else {
                                    trade.exit_bar_index
                                };
                                
                                let start_pos = start_bar as i32 - leftmost_bar_position;
                                let end_pos = end_bar as i32 - leftmost_bar_position;
                                
                                if end_pos >= 0 && start_pos < bars_that_fit {
                                    let seg_start_x = chart_rect.left() + start_pos as f32 * bar_width;
                                    let seg_end_x = chart_rect.left() + end_pos as f32 * bar_width;
                                    let avg_y = chart_rect.top() + (1.0 - (avg_price - min_price) / price_range) as f32 * chart_rect.height();
                                    
                                    // Create a quad from average price to exit price
                                    let points = vec![
                                        egui::pos2(seg_start_x.max(chart_rect.left()), avg_y),
                                        egui::pos2(seg_end_x.min(chart_rect.right()), avg_y),
                                        egui::pos2(seg_end_x.min(chart_rect.right()), exit_y),
                                        egui::pos2(seg_start_x.max(chart_rect.left()), exit_y),
                                    ];
                                    
                                    painter.add(egui::Shape::convex_polygon(points, fill_color, egui::Stroke::NONE));
                                }
                            }
                            
                            // Draw P&L text above the trade
                            let mid_x = (entry_x + exit_x) / 2.0;
                            let text_y = entry_y.min(exit_y) - 20.0;
                            let pnl_text = format!("${:.2}", trade.pnl);
                            let text_color = if trade.pnl >= 0.0 {
                                egui::Color32::GREEN
                            } else {
                                egui::Color32::RED
                            };
                            
                            painter.text(
                                egui::pos2(mid_x, text_y),
                                egui::Align2::CENTER_BOTTOM,
                                pnl_text,
                                egui::FontId::default(),
                                text_color,
                            );
                        }
                    }
                    
                    
                    // Draw bars at their positions
                    for (i, bar) in visible_data.iter().enumerate() {
                        let _bar_index = start_idx + i;
                        // Position bars based on their index relative to leftmost_bar_position
                        let bar_position = (start_idx as i32 + i as i32 - leftmost_bar_position) as f32;
                        let x = chart_rect.left() + bar_position * bar_width + bar_width / 2.0;
                        
                        // Skip bars that would be drawn outside the chart area
                        if x < chart_rect.left() - bar_width || x > chart_rect.right() + bar_width {
                            continue;
                        }
                        
                        
                        let high_y = chart_rect.top() + (1.0 - (bar.high - min_price) / price_range) as f32 * chart_rect.height();
                        let low_y = chart_rect.top() + (1.0 - (bar.low - min_price) / price_range) as f32 * chart_rect.height();
                        let open_y = chart_rect.top() + (1.0 - (bar.open - min_price) / price_range) as f32 * chart_rect.height();
                        let close_y = chart_rect.top() + (1.0 - (bar.close - min_price) / price_range) as f32 * chart_rect.height();
                        
                        // Dim colors for non-trading hours (use stock market hours)
                        let is_trading = is_stock_market_hours(&bar.timestamp);
                        
                        let color = if bar.close >= bar.open { 
                            if is_trading {
                                egui::Color32::from_rgb(0, 255, 0)
                            } else {
                                egui::Color32::from_rgba_unmultiplied(0, 255, 0, 100)
                            }
                        } else { 
                            if is_trading {
                                egui::Color32::from_rgb(255, 0, 0)
                            } else {
                                egui::Color32::from_rgba_unmultiplied(255, 0, 0, 100)
                            }
                        };
                        
                        // Draw high-low line
                        painter.line_segment(
                            [egui::pos2(x, high_y), egui::pos2(x, low_y)],
                            egui::Stroke::new(1.0, color),
                        );
                        
                        // Draw open-close body
                        let body_rect = egui::Rect::from_x_y_ranges(
                            (x - bar_width * 0.3)..=(x + bar_width * 0.3),
                            open_y.min(close_y)..=open_y.max(close_y),
                        );
                        painter.rect_filled(body_rect, 0.0, color);
                    }
                    
                    
                    // Draw average entry price lines showing history
                    if !self.average_price_history.is_empty() {
                        let mut current_avg_price = 0.0;
                        let mut current_start_idx = 0;
                        let mut is_long = true;
                        
                        for &(bar_idx, avg_price) in &self.average_price_history {
                            // Draw the previous segment if we're starting a new one or closing
                            if current_avg_price != 0.0 && (avg_price == 0.0 || (avg_price != current_avg_price && bar_idx > current_start_idx)) {
                                let start_x_idx = if current_start_idx >= start_idx { 
                                    current_start_idx - start_idx 
                                } else { 
                                    0 
                                };
                                let end_x_idx = if bar_idx >= start_idx { 
                                    bar_idx - start_idx 
                                } else { 
                                    0 
                                };
                                
                                if end_x_idx > 0 && start_x_idx < end_x_idx {
                                    let avg_y = chart_rect.top() + (1.0 - (current_avg_price - min_price) / price_range) as f32 * chart_rect.height();
                                    // Position based on actual bar index relative to leftmost_bar_position
                                    let start_bar_pos = (current_start_idx as i32 - leftmost_bar_position).max(0) as f32;
                                    let end_bar_pos = (bar_idx as i32 - leftmost_bar_position).max(0) as f32;
                                    let start_x = chart_rect.left() + start_bar_pos * bar_width;
                                    let end_x = chart_rect.left() + end_bar_pos * bar_width;
                                    
                                    let color = if is_long {
                                        egui::Color32::from_rgb(0, 200, 0)
                                    } else {
                                        egui::Color32::from_rgb(200, 0, 0)
                                    };
                                    
                                    // Draw dashed line
                                    let dash_length = 10.0;
                                    let gap_length = 5.0;
                                    let mut x = start_x;
                                    while x < end_x {
                                        let dash_end = (x + dash_length).min(end_x);
                                        painter.line_segment(
                                            [egui::pos2(x, avg_y), egui::pos2(dash_end, avg_y)],
                                            egui::Stroke::new(2.0, color),
                                        );
                                        x += dash_length + gap_length;
                                    }
                                }
                            }
                            
                            // Only update if not a closing marker
                            if avg_price != 0.0 {
                                current_avg_price = avg_price;
                                current_start_idx = bar_idx;
                                // Determine if position is long or short based on transaction at this bar
                                if let Some(transaction) = self.transactions.iter().find(|t| 
                                    self.bars[bar_idx].timestamp == t.timestamp
                                ) {
                                    is_long = transaction.position_after > 0;
                                }
                            } else {
                                // Reset for next position
                                current_avg_price = 0.0;
                            }
                        }
                        
                        // Draw the current segment if position is still open
                        if self.position != 0 && current_avg_price > 0.0 {
                            let _start_x_idx = if current_start_idx >= start_idx { 
                                current_start_idx - start_idx 
                            } else { 
                                0 
                            };
                            
                            if true {
                                let avg_y = chart_rect.top() + (1.0 - (current_avg_price - min_price) / price_range) as f32 * chart_rect.height();
                                // Position based on actual bar index relative to leftmost_bar_position
                                let start_bar_pos = (current_start_idx as i32 - leftmost_bar_position).max(0) as f32;
                                let end_bar_pos = (self.current_bar_index as i32 - leftmost_bar_position).max(0) as f32;
                                let start_x = chart_rect.left() + start_bar_pos * bar_width;
                                let end_x = chart_rect.left() + end_bar_pos * bar_width;
                                
                                let color = if self.position > 0 {
                                    egui::Color32::from_rgb(0, 200, 0)
                                } else {
                                    egui::Color32::from_rgb(200, 0, 0)
                                };
                                
                                // Draw dashed line - respect right margin
                                let dash_length = 10.0;
                                let gap_length = 5.0;
                                let mut x = start_x;
                                while x < end_x {
                                    let dash_end = (x + dash_length).min(end_x);
                                    painter.line_segment(
                                        [egui::pos2(x, avg_y), egui::pos2(dash_end, avg_y)],
                                        egui::Stroke::new(2.0, color),
                                    );
                                    x += dash_length + gap_length;
                                }
                                
                                // Draw price label with unrealized P&L
                                let unrealized = self.calculate_unrealized_pnl();
                                let text = format!("Avg: {:.5} | Unrealized: ${:.2}", current_avg_price, unrealized);
                                let galley = painter.layout_no_wrap(
                                    text,
                                    egui::FontId::default(),
                                    color,
                                );
                                painter.galley(egui::pos2(start_x + 5.0, avg_y - galley.size().y - 2.0), galley, color);
                                
                                // Draw stop loss and take profit lines if active
                                if self.position != 0 && current_avg_price > 0.0 {
                                    let current_price = self.bars[self.current_bar_index].close;
                                    
                                    // Calculate P&L per unit price move
                                    let (_, pnl_per_unit) = self.calculate_pnl(current_price + 0.0001, self.position.abs(), self.position > 0);
                                    let (_, base_pnl) = self.calculate_pnl(current_price, self.position.abs(), self.position > 0);
                                    let pnl_per_pip = (pnl_per_unit - base_pnl) * 10000.0; // P&L per pip/point
                                    
                                    // Stop Loss Line
                                    if let Some(stop_loss) = self.stop_loss_dollars {
                                        let sl_price = if self.position > 0 {
                                            current_avg_price - (stop_loss / pnl_per_pip * 0.0001)
                                        } else {
                                            current_avg_price + (stop_loss / pnl_per_pip * 0.0001)
                                        };
                                        
                                        if sl_price >= min_price && sl_price <= max_price {
                                            let sl_y = chart_rect.top() + (1.0 - (sl_price - min_price) / price_range) as f32 * chart_rect.height();
                                            
                                            // Draw dotted red line
                                            let dot_length = 5.0;
                                            let gap_length = 5.0;
                                            let mut x = start_x;
                                            while x < end_x {
                                                let dot_end = (x + dot_length).min(end_x);
                                                painter.line_segment(
                                                    [egui::pos2(x, sl_y), egui::pos2(dot_end, sl_y)],
                                                    egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 100, 100)),
                                                );
                                                x += dot_length + gap_length;
                                            }
                                            
                                            // Label
                                            let sl_text = format!("SL: {:.5} (-${:.0})", sl_price, stop_loss);
                                            painter.text(
                                                egui::pos2(end_x + 5.0, sl_y - 5.0),
                                                egui::Align2::LEFT_CENTER,
                                                sl_text,
                                                egui::FontId::default(),
                                                egui::Color32::from_rgb(255, 100, 100),
                                            );
                                        }
                                    }
                                    
                                    // Take Profit Line
                                    if let Some(take_profit) = self.take_profit_dollars {
                                        let tp_price = if self.position > 0 {
                                            current_avg_price + (take_profit / pnl_per_pip * 0.0001)
                                        } else {
                                            current_avg_price - (take_profit / pnl_per_pip * 0.0001)
                                        };
                                        
                                        if tp_price >= min_price && tp_price <= max_price {
                                            let tp_y = chart_rect.top() + (1.0 - (tp_price - min_price) / price_range) as f32 * chart_rect.height();
                                            
                                            // Draw dotted green line
                                            let dot_length = 5.0;
                                            let gap_length = 5.0;
                                            let mut x = start_x;
                                            while x < end_x {
                                                let dot_end = (x + dot_length).min(end_x);
                                                painter.line_segment(
                                                    [egui::pos2(x, tp_y), egui::pos2(dot_end, tp_y)],
                                                    egui::Stroke::new(1.5, egui::Color32::from_rgb(100, 255, 100)),
                                                );
                                                x += dot_length + gap_length;
                                            }
                                            
                                            // Label
                                            let tp_text = format!("TP: {:.5} (+${:.0})", tp_price, take_profit);
                                            painter.text(
                                                egui::pos2(end_x + 5.0, tp_y - 5.0),
                                                egui::Align2::LEFT_CENTER,
                                                tp_text,
                                                egui::FontId::default(),
                                                egui::Color32::from_rgb(100, 255, 100),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    // Draw Strategy Indicators
                    if self.enable_strategy_signals && self.show_regime_indicator {
                        let strategy = self.get_current_strategy();
                        
                        if strategy == TradingStrategy::Momentum && start_idx + 20 <= end_idx {
                            // Draw 20-bar SMA for momentum strategy
                            let mut points = Vec::new();
                            
                            for i in start_idx..=end_idx.min(self.bars.len() - 1) {
                                if i >= 19 {
                                    let mut sum = 0.0;
                                    for j in (i - 19)..=i {
                                        sum += self.bars[j].close;
                                    }
                                    let sma = sum / 20.0;
                                    
                                    let bar_position = (i as i32 - leftmost_bar_position) as f32;
                                    let x = chart_rect.left() + bar_position * bar_width + bar_width / 2.0;
                                    let y = chart_rect.bottom() - ((sma - min_price) / price_range * chart_rect.height() as f64) as f32;
                                    
                                    if x >= chart_rect.left() && x <= chart_rect.right() {
                                        points.push(egui::pos2(x, y));
                                    }
                                }
                            }
                            
                            // Draw SMA line
                            for i in 1..points.len() {
                                painter.line_segment(
                                    [points[i-1], points[i]],
                                    egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 165, 0))
                                );
                            }
                        }
                        
                        if strategy == TradingStrategy::MeanRevert && start_idx + 20 <= end_idx {
                            // Draw Bollinger Bands for mean revert strategy
                            let mut upper_points = Vec::new();
                            let mut lower_points = Vec::new();
                            let mut middle_points = Vec::new();
                            
                            for i in start_idx..=end_idx.min(self.bars.len() - 1) {
                                if i >= 19 {
                                    // Calculate 20-bar mean
                                    let mut sum = 0.0;
                                    for j in (i - 19)..=i {
                                        sum += self.bars[j].close;
                                    }
                                    let mean = sum / 20.0;
                                    
                                    // Calculate std dev
                                    let mut variance_sum = 0.0;
                                    for j in (i - 19)..=i {
                                        let diff = self.bars[j].close - mean;
                                        variance_sum += diff * diff;
                                    }
                                    let std_dev = (variance_sum / 20.0).sqrt();
                                    
                                    let upper = mean + 2.0 * std_dev;
                                    let lower = mean - 2.0 * std_dev;
                                    
                                    let bar_position = (i as i32 - leftmost_bar_position) as f32;
                                    let x = chart_rect.left() + bar_position * bar_width + bar_width / 2.0;
                                    
                                    if x >= chart_rect.left() && x <= chart_rect.right() {
                                        let y_upper = chart_rect.bottom() - ((upper - min_price) / price_range * chart_rect.height() as f64) as f32;
                                        let y_lower = chart_rect.bottom() - ((lower - min_price) / price_range * chart_rect.height() as f64) as f32;
                                        let y_middle = chart_rect.bottom() - ((mean - min_price) / price_range * chart_rect.height() as f64) as f32;
                                        
                                        upper_points.push(egui::pos2(x, y_upper));
                                        lower_points.push(egui::pos2(x, y_lower));
                                        middle_points.push(egui::pos2(x, y_middle));
                                    }
                                }
                            }
                            
                            // Draw bands
                            let band_color = egui::Color32::from_rgba_unmultiplied(100, 150, 255, 100);
                            for i in 1..upper_points.len() {
                                painter.line_segment(
                                    [upper_points[i-1], upper_points[i]],
                                    egui::Stroke::new(1.0, band_color)
                                );
                                painter.line_segment(
                                    [lower_points[i-1], lower_points[i]],
                                    egui::Stroke::new(1.0, band_color)
                                );
                                painter.line_segment(
                                    [middle_points[i-1], middle_points[i]],
                                    egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(100, 150, 255, 150))
                                );
                            }
                        }
                    }
                    
                    // Draw Linear Regression Channel
                    if self.show_linreg && self.current_bar_index >= self.linreg_period {
                        if let Some((intercept, slope, std_dev)) = self.calculate_linear_regression(self.current_bar_index, self.linreg_period) {
                            // Draw regression line and channel
                            let lookback_start = self.current_bar_index.saturating_sub(self.linreg_period - 1);
                            
                            for i in 0..self.linreg_period {
                                let bar_idx = lookback_start + i;
                                if bar_idx >= start_idx && bar_idx < end_idx && i > 0 {
                                    let prev_idx = bar_idx - 1;
                                    
                                    // Calculate regression values
                                    let reg_value = intercept + slope * i as f64;
                                    let prev_reg_value = intercept + slope * (i - 1) as f64;
                                    
                                    // Calculate positions
                                    let bar_position = (bar_idx as i32 - leftmost_bar_position) as f32;
                                    let prev_bar_position = (prev_idx as i32 - leftmost_bar_position) as f32;
                                    let x = chart_rect.left() + bar_position * bar_width + bar_width / 2.0;
                                    let x_prev = chart_rect.left() + prev_bar_position * bar_width + bar_width / 2.0;
                                    
                                    // Main regression line
                                    let y = chart_rect.top() + (1.0 - (reg_value - min_price) / price_range) as f32 * chart_rect.height();
                                    let y_prev = chart_rect.top() + (1.0 - (prev_reg_value - min_price) / price_range) as f32 * chart_rect.height();
                                    
                                    // Color based on slope
                                    let line_color = if slope > 0.0 {
                                        egui::Color32::from_rgb(0, 255, 0) // Green for positive slope
                                    } else if slope < 0.0 {
                                        egui::Color32::from_rgb(255, 0, 0) // Red for negative slope  
                                    } else {
                                        egui::Color32::from_rgb(255, 255, 0) // Yellow for flat
                                    };
                                    
                                    painter.line_segment(
                                        [egui::pos2(x_prev, y_prev), egui::pos2(x, y)],
                                        egui::Stroke::new(2.0, line_color),
                                    );
                                    
                                    // Draw channel lines (1 standard deviation)
                                    let upper = reg_value + std_dev;
                                    let lower = reg_value - std_dev;
                                    let prev_upper = prev_reg_value + std_dev;
                                    let prev_lower = prev_reg_value - std_dev;
                                    
                                    let y_upper = chart_rect.top() + (1.0 - (upper - min_price) / price_range) as f32 * chart_rect.height();
                                    let y_lower = chart_rect.top() + (1.0 - (lower - min_price) / price_range) as f32 * chart_rect.height();
                                    let y_prev_upper = chart_rect.top() + (1.0 - (prev_upper - min_price) / price_range) as f32 * chart_rect.height();
                                    let y_prev_lower = chart_rect.top() + (1.0 - (prev_lower - min_price) / price_range) as f32 * chart_rect.height();
                                    
                                    let channel_color = egui::Color32::from_rgba_unmultiplied(200, 200, 200, 100);
                                    
                                    painter.line_segment(
                                        [egui::pos2(x_prev, y_prev_upper), egui::pos2(x, y_upper)],
                                        egui::Stroke::new(1.0, channel_color),
                                    );
                                    
                                    painter.line_segment(
                                        [egui::pos2(x_prev, y_prev_lower), egui::pos2(x, y_lower)],
                                        egui::Stroke::new(1.0, channel_color),
                                    );
                                }
                            }
                            
                            // Show slope info
                            if slope != 0.0 {
                                let slope_text = format!("Slope: {:.6}", slope);
                                painter.text(
                                    egui::pos2(chart_rect.right() - 100.0, chart_rect.top() + 20.0),
                                    egui::Align2::RIGHT_TOP,
                                    slope_text,
                                    egui::FontId::default(),
                                    if slope > 0.0 { egui::Color32::GREEN } else { egui::Color32::RED },
                                );
                            }
                        }
                    }
                    
                    // Draw T3 indicator line - HIDDEN
                    /*
                    if !self.t3_values.is_empty() && self.current_bar_index < self.t3_values.len() {
                        let t3_color = egui::Color32::from_rgb(255, 165, 0); // Orange color for T3
                        
                        // Draw T3 line connecting visible bars
                        for i in 0..visible_data.len() {
                            let bar_idx = start_idx + i;
                            
                            if bar_idx > 0 && bar_idx < self.t3_values.len() {
                                let t3_curr = self.t3_values[bar_idx];
                                let t3_prev = self.t3_values[bar_idx - 1];
                                
                                // Use same positioning as bars
                                let bar_position_curr = (bar_idx as i32 - leftmost_bar_position) as f32;
                                let bar_position_prev = ((bar_idx - 1) as i32 - leftmost_bar_position) as f32;
                                
                                let x_curr = chart_rect.left() + bar_position_curr * bar_width + bar_width / 2.0;
                                let x_prev = chart_rect.left() + bar_position_prev * bar_width + bar_width / 2.0;
                                
                                let y_curr = chart_rect.bottom() - ((t3_curr - min_price) / price_range * chart_rect.height() as f64) as f32;
                                let y_prev = chart_rect.bottom() - ((t3_prev - min_price) / price_range * chart_rect.height() as f64) as f32;
                                
                                // Only draw if both points are within the visible area
                                if x_prev >= chart_rect.left() - bar_width && x_curr <= chart_rect.right() + bar_width {
                                    painter.line_segment(
                                        [egui::pos2(x_prev, y_prev), egui::pos2(x_curr, y_curr)],
                                        egui::Stroke::new(2.0, t3_color),
                                    );
                                }
                            }
                        }
                    }
                    */
                    
                    // Draw linear regression channel on the past 50 bars
                    // COMMENTED OUT - Linear regression channel is currently hidden
                    /*
                    let regression_period = 50;
                    if self.current_bar_index >= regression_period {
                        if let Some((intercept, slope, std_dev)) = self.calculate_linear_regression(self.current_bar_index, regression_period) {
                            let channel_start_bar = self.current_bar_index + 1 - regression_period;
                            
                            // Draw the regression line and channel
                            let mut regression_points = Vec::new();
                            let mut upper_channel_points = Vec::new();
                            let mut lower_channel_points = Vec::new();
                            
                            for i in 0..regression_period {
                                let bar_index = channel_start_bar + i;
                                let bar_position = (bar_index as i32 - leftmost_bar_position) as f32;
                                
                                // Only draw if visible
                                if bar_position >= -1.0 && bar_position <= bars_that_fit as f32 + 1.0 {
                                    let x = chart_rect.left() + bar_position * bar_width + bar_width / 2.0;
                                    
                                    // Calculate y values for regression line and channels
                                    let regression_value = intercept + slope * i as f64;
                                    let upper_value = regression_value + 2.0 * std_dev;
                                    let lower_value = regression_value - 2.0 * std_dev;
                                    
                                    let reg_y = chart_rect.top() + (1.0 - (regression_value - min_price) / price_range) as f32 * chart_rect.height();
                                    let upper_y = chart_rect.top() + (1.0 - (upper_value - min_price) / price_range) as f32 * chart_rect.height();
                                    let lower_y = chart_rect.top() + (1.0 - (lower_value - min_price) / price_range) as f32 * chart_rect.height();
                                    
                                    regression_points.push(egui::pos2(x, reg_y));
                                    upper_channel_points.push(egui::pos2(x, upper_y));
                                    lower_channel_points.push(egui::pos2(x, lower_y));
                                }
                            }
                            
                            // Draw channel lines
                            let channel_color = egui::Color32::from_rgba_unmultiplied(100, 150, 200, 100);
                            let center_color = egui::Color32::from_rgba_unmultiplied(150, 200, 250, 150);
                            
                            // Draw upper channel
                            for i in 1..upper_channel_points.len() {
                                painter.line_segment(
                                    [upper_channel_points[i-1], upper_channel_points[i]],
                                    egui::Stroke::new(1.0, channel_color),
                                );
                            }
                            
                            // Draw lower channel
                            for i in 1..lower_channel_points.len() {
                                painter.line_segment(
                                    [lower_channel_points[i-1], lower_channel_points[i]],
                                    egui::Stroke::new(1.0, channel_color),
                                );
                            }
                            
                            // Draw center regression line
                            for i in 1..regression_points.len() {
                                painter.line_segment(
                                    [regression_points[i-1], regression_points[i]],
                                    egui::Stroke::new(2.0, center_color),
                                );
                            }
                        }
                    }
                    */
                }
                
                // Draw timestamp in bottom right of chart
                if self.current_bar_index < self.bars.len() {
                    let bar_time = self.bars[self.current_bar_index].timestamp.with_timezone(&self.display_timezone);
                    let timestamp_text = bar_time.format("%Y-%m-%d %H:%M:%S EST").to_string();
                    let galley = painter.layout_no_wrap(
                        timestamp_text,
                        egui::FontId::default(),
                        egui::Color32::WHITE,
                    );
                    let text_pos = egui::pos2(
                        chart_rect.right() - galley.size().x - 10.0,
                        chart_rect.bottom() - galley.size().y - 10.0,
                    );
                    // Draw background for better visibility
                    painter.rect_filled(
                        egui::Rect::from_min_size(
                            text_pos - egui::vec2(5.0, 2.0),
                            galley.size() + egui::vec2(10.0, 4.0),
                        ),
                        4.0,
                        egui::Color32::from_rgba_premultiplied(0, 0, 0, 180),
                    );
                    painter.galley(text_pos, galley, egui::Color32::WHITE);
                }
                
                }
            });
            
            
            // MACD Panel
            // Only show MACD panel if enabled
            if self.show_macd {
                ui.group(|ui| {
                    // Use all available width with small margins
                    let chart_width = available_size.x.max(100.0);
                    let macd_size = egui::Vec2::new(chart_width, macd_height);
                    let (response, painter) = ui.allocate_painter(macd_size, egui::Sense::hover());
                    let macd_rect = response.rect;
                    
                    // Draw MACD background
                    painter.rect_filled(macd_rect, 0.0, egui::Color32::from_gray(20));
                    
                    if !self.bars.is_empty() && self.current_bar_index >= self.t3_slow_length - 1 && !self.macd_values.is_empty() {
                    // Calculate bar positioning using same logic as main chart
                    let bar_width = BASE_BAR_WIDTH * self.zoom_level;
                    let bars_that_fit = (chart_width / bar_width) as usize;
                    let leftmost_bar_position = (self.current_bar_index as i32 + self.view_offset - bars_that_fit as i32 + 1).max(0);
                    
                    let start_idx = leftmost_bar_position.max(0) as usize;
                    let end_idx = (leftmost_bar_position + bars_that_fit as i32).min(self.current_bar_index as i32 + 1).max(0) as usize;
                    
                    // Find min/max MACD values for visible range
                    let mut macd_min = 0.0f64;
                    let mut macd_max = 0.0f64;
                    
                    for i in start_idx..end_idx {
                        if i >= 19 {
                            let macd_idx = i - 19;
                            if macd_idx < self.macd_values.len() {
                                let macd = self.macd_values[macd_idx];
                                let signal = self.macd_signal.get(macd_idx).copied().unwrap_or(macd);
                                let histogram = self.macd_histogram.get(macd_idx).copied().unwrap_or(0.0);
                                
                                macd_min = macd_min.min(macd).min(signal).min(histogram);
                                macd_max = macd_max.max(macd).max(signal).max(histogram);
                            }
                        }
                    }
                    
                    // Add padding
                    let padding = (macd_max - macd_min).max(0.0001) * 0.1;
                    macd_min -= padding;
                    macd_max += padding;
                    let macd_range = macd_max - macd_min;
                    
                    if macd_range > 0.0 {
                        // Draw zero line
                        let zero_y = macd_rect.bottom() - ((0.0 - macd_min) / macd_range * macd_rect.height() as f64) as f32;
                        painter.line_segment(
                            [egui::pos2(macd_rect.left(), zero_y), egui::pos2(macd_rect.right(), zero_y)],
                            egui::Stroke::new(1.0, egui::Color32::from_gray(80)),
                        );
                        
                        // Draw histogram bars first (so they appear behind lines)
                        for i in start_idx..end_idx {
                            if i >= 19 {
                                let macd_idx = i - 19;
                                if macd_idx < self.macd_histogram.len() {
                                    let histogram = self.macd_histogram[macd_idx];
                                    let bar_position = (i as i32 - leftmost_bar_position) as f32;
                                    let x = macd_rect.left() + bar_position * bar_width + bar_width / 2.0;
                                    
                                    let hist_y = macd_rect.bottom() - ((histogram - macd_min) / macd_range * macd_rect.height() as f64) as f32;
                                    
                                    let color = if histogram >= 0.0 {
                                        egui::Color32::from_rgba_unmultiplied(0, 200, 0, 120)
                                    } else {
                                        egui::Color32::from_rgba_unmultiplied(200, 0, 0, 120)
                                    };
                                    
                                    let bar_rect = egui::Rect::from_x_y_ranges(
                                        (x - bar_width * 0.3)..=(x + bar_width * 0.3),
                                        zero_y.min(hist_y)..=zero_y.max(hist_y),
                                    );
                                    painter.rect_filled(bar_rect, 0.0, color);
                                }
                            }
                        }
                        
                        // Draw MACD and Signal lines
                        for i in (start_idx + 1)..end_idx {
                            if i >= 19 && i > 0 {
                                let macd_idx = i - 19;
                                let macd_idx_prev = (i - 1) - 19;
                                
                                if macd_idx < self.macd_values.len() && macd_idx_prev < self.macd_values.len() {
                                    let bar_position = (i as i32 - leftmost_bar_position) as f32;
                                    let bar_position_prev = ((i - 1) as i32 - leftmost_bar_position) as f32;
                                    
                                    let x = macd_rect.left() + bar_position * bar_width + bar_width / 2.0;
                                    let x_prev = macd_rect.left() + bar_position_prev * bar_width + bar_width / 2.0;
                                    
                                    // MACD line
                                    let macd_curr = self.macd_values[macd_idx];
                                    let macd_prev = self.macd_values[macd_idx_prev];
                                    
                                    let y_curr = macd_rect.bottom() - ((macd_curr - macd_min) / macd_range * macd_rect.height() as f64) as f32;
                                    let y_prev = macd_rect.bottom() - ((macd_prev - macd_min) / macd_range * macd_rect.height() as f64) as f32;
                                    
                                    painter.line_segment(
                                        [egui::pos2(x_prev, y_prev), egui::pos2(x, y_curr)],
                                        egui::Stroke::new(2.0, egui::Color32::from_rgb(100, 149, 237)), // Blue
                                    );
                                    
                                    // Signal line
                                    if macd_idx < self.macd_signal.len() && macd_idx_prev < self.macd_signal.len() {
                                        let signal_curr = self.macd_signal[macd_idx];
                                        let signal_prev = self.macd_signal[macd_idx_prev];
                                        
                                        let y_curr_signal = macd_rect.bottom() - ((signal_curr - macd_min) / macd_range * macd_rect.height() as f64) as f32;
                                        let y_prev_signal = macd_rect.bottom() - ((signal_prev - macd_min) / macd_range * macd_rect.height() as f64) as f32;
                                        
                                        painter.line_segment(
                                            [egui::pos2(x_prev, y_prev_signal), egui::pos2(x, y_curr_signal)],
                                            egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 165, 0)), // Orange
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            });
            } // End if self.show_macd
            
            ui.separator();
            
            // Transaction log as a table
            let mut copy_csv = false;
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.label("Transaction Log:");
                    if ui.button("📋 Copy as CSV").clicked() {
                        copy_csv = true;
                    }
                });
                let log_height = ui.available_height().min(200.0);
                
                egui::ScrollArea::vertical()
                    .max_height(log_height)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        use egui_extras::{TableBuilder, Column};
                        
                        TableBuilder::new(ui)
                            .striped(true)
                            .resizable(true)
                            .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
                            .column(Column::auto().at_least(150.0)) // Timestamp
                            .column(Column::auto().at_least(50.0))  // Action
                            .column(Column::auto().at_least(50.0))  // Lots
                            .column(Column::auto().at_least(80.0))  // Price
                            .column(Column::auto().at_least(50.0))  // Position
                            .column(Column::auto().at_least(80.0))  // Gross P&L
                            .column(Column::auto().at_least(80.0))  // Net P&L
                            .column(Column::auto().at_least(80.0))  // Total
                            .column(Column::auto().at_least(80.0))  // MaxDD
                            .column(Column::auto().at_least(80.0))  // Equity
                            .header(20.0, |mut header| {
                                header.col(|ui| { ui.strong("Timestamp"); });
                                header.col(|ui| { ui.strong("Action"); });
                                header.col(|ui| { ui.strong("Lots"); });
                                header.col(|ui| { ui.strong("Price"); });
                                header.col(|ui| { ui.strong("Pos"); });
                                header.col(|ui| { ui.strong("Gross P&L"); });
                                header.col(|ui| { ui.strong("Net P&L"); });
                                header.col(|ui| { ui.strong("Total"); });
                                header.col(|ui| { ui.strong("MaxDD"); });
                                header.col(|ui| { ui.strong("Tied Up"); });
                            })
                            .body(|mut body| {
                                for transaction in self.transactions.iter().rev() {
                                    let pnl_color = if transaction.trade_pnl >= 0.0 {
                                        egui::Color32::GREEN
                                    } else {
                                        egui::Color32::RED
                                    };
                                    
                                    body.row(18.0, |mut row| {
                                        row.col(|ui| {
                                            let text = format!("{}", transaction.timestamp.with_timezone(&self.display_timezone).format("%Y-%m-%d %H:%M:%S"));
                                            ui.add(egui::Label::new(&text).sense(egui::Sense::click()));
                                            if ui.input(|i| i.pointer.any_click()) {
                                                ui.output_mut(|o| o.copied_text = text);
                                            }
                                        });
                                        row.col(|ui| {
                                            let text = format!("{:?}", transaction.action);
                                            ui.add(egui::Label::new(&text).sense(egui::Sense::click()));
                                            if ui.input(|i| i.pointer.any_click()) {
                                                ui.output_mut(|o| o.copied_text = text);
                                            }
                                        });
                                        row.col(|ui| {
                                            let text = format!("{}", transaction.lots);
                                            ui.add(egui::Label::new(&text).sense(egui::Sense::click()));
                                            if ui.input(|i| i.pointer.any_click()) {
                                                ui.output_mut(|o| o.copied_text = text);
                                            }
                                        });
                                        row.col(|ui| {
                                            let text = format!("{:.5}", transaction.price);
                                            ui.add(egui::Label::new(&text).sense(egui::Sense::click()));
                                            if ui.input(|i| i.pointer.any_click()) {
                                                ui.output_mut(|o| o.copied_text = text);
                                            }
                                        });
                                        row.col(|ui| {
                                            let text = format!("{}", transaction.position_after);
                                            ui.add(egui::Label::new(&text).sense(egui::Sense::click()));
                                            if ui.input(|i| i.pointer.any_click()) {
                                                ui.output_mut(|o| o.copied_text = text);
                                            }
                                        });
                                        row.col(|ui| {
                                            let text = format!("${:.2}", transaction.gross_pnl);
                                            ui.add(egui::Label::new(&text).sense(egui::Sense::click()));
                                            if ui.input(|i| i.pointer.any_click()) {
                                                ui.output_mut(|o| o.copied_text = text);
                                            }
                                        });
                                        row.col(|ui| {
                                            let text = format!("${:.2}", transaction.trade_pnl);
                                            ui.add(egui::Label::new(egui::RichText::new(&text).color(pnl_color)).sense(egui::Sense::click()));
                                            if ui.input(|i| i.pointer.any_click()) {
                                                ui.output_mut(|o| o.copied_text = text);
                                            }
                                        });
                                        row.col(|ui| {
                                            let text = format!("${:.2}", transaction.cumulative_pnl);
                                            ui.add(egui::Label::new(&text).sense(egui::Sense::click()));
                                            if ui.input(|i| i.pointer.any_click()) {
                                                ui.output_mut(|o| o.copied_text = text);
                                            }
                                        });
                                        row.col(|ui| {
                                            if transaction.max_unrealized_drawdown > 0.0 {
                                                let text = format!("${:.2}", transaction.max_unrealized_drawdown);
                                                ui.add(egui::Label::new(&text).sense(egui::Sense::click()));
                                                if ui.input(|i| i.pointer.any_click()) {
                                                    ui.output_mut(|o| o.copied_text = text);
                                                }
                                            }
                                        });
                                        row.col(|ui| {
                                            if transaction.position_after != 0 {
                                                let text = format!("${:.2}", transaction.equity_tied_up);
                                                ui.add(egui::Label::new(&text).sense(egui::Sense::click()));
                                                if ui.input(|i| i.pointer.any_click()) {
                                                    ui.output_mut(|o| o.copied_text = text);
                                                }
                                            }
                                        });
                                    });
                                }
                            });
                });
            });
            
            // Handle CSV copy outside of the table
            if copy_csv {
                self.copy_transactions_as_csv(ui);
            }
        });

        // Request repaint for smooth playback and window resizing
        if self.playing {
            ctx.request_repaint();
        } else {
            // Also request repaint on resize/zoom changes
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
}

fn is_trading_hours(timestamp: &DateTime<Utc>) -> bool {
    use chrono::{Weekday, FixedOffset};
    
    // Convert to Eastern Time (EST/EDT)
    // EST = UTC-5, EDT = UTC-4
    // Using EST offset for now (-5 hours = -18000 seconds)
    let est_offset = FixedOffset::west_opt(5 * 3600).unwrap();
    let est_time = timestamp.with_timezone(&est_offset);
    
    let weekday = est_time.weekday();
    let hour = est_time.hour();
    
    // Forex market hours: Sunday 5PM - Friday 5PM EST
    match weekday {
        Weekday::Sat => false,
        Weekday::Sun => hour >= 17, // 5PM Sunday
        Weekday::Fri => hour < 17,  // Until 5PM Friday
        _ => true,
    }
}

fn is_stock_market_hours(timestamp: &DateTime<Utc>) -> bool {
    use chrono::{Weekday, FixedOffset};
    
    // Convert to Eastern Time (EST/EDT)
    let est_offset = FixedOffset::west_opt(5 * 3600).unwrap();
    let est_time = timestamp.with_timezone(&est_offset);
    
    let weekday = est_time.weekday();
    let hour = est_time.hour();
    let minute = est_time.minute();
    
    // Stock market hours: Monday-Friday 9:30 AM - 4:00 PM EST
    match weekday {
        Weekday::Sat | Weekday::Sun => false,
        _ => {
            // Market open at 9:30 AM (hour=9, minute>=30) or after 10 AM
            let after_open = hour > 9 || (hour == 9 && minute >= 30);
            // Market closes at 4:00 PM
            let before_close = hour < 16;
            after_open && before_close
        }
    }
}

fn is_market_close(timestamp: &DateTime<Utc>, next_timestamp: Option<&DateTime<Utc>>) -> bool {
    use chrono::{FixedOffset, Weekday};
    
    // Convert to Eastern Time
    let est_offset = FixedOffset::west_opt(5 * 3600).unwrap();
    let est_time = timestamp.with_timezone(&est_offset);
    
    // Skip weekends
    match est_time.weekday() {
        Weekday::Sat | Weekday::Sun => return false,
        _ => {}
    }
    
    let hour = est_time.hour();
    let minute = est_time.minute();
    
    // Check if current time is at or after 3:50 PM (close positions before 4 PM with buffer)
    if hour == 15 && minute >= 50 {
        return true;
    }
    if hour >= 16 {
        return true;
    }
    
    // Also check if we're before market open (shouldn't have overnight positions)
    if hour < 9 || (hour == 9 && minute < 30) {
        return true;
    }
    
    // If we have a next timestamp, check various conditions
    if let Some(next_ts) = next_timestamp {
        let next_est = next_ts.with_timezone(&est_offset);
        
        // If next bar is on a different day, we're at end of trading day
        if next_est.date_naive() != est_time.date_naive() {
            return true;
        }
        
        // If next bar is a weekend, we're at end of trading week
        match next_est.weekday() {
            Weekday::Sat | Weekday::Sun => return true,
            _ => {}
        }
        
        // If current is before 3:55 PM and next is at or after 3:55 PM
        if hour == 15 && minute < 55 
            && (next_est.hour() > 15 || (next_est.hour() == 15 && next_est.minute() >= 55)) {
            return true;
        }
        
        // If there's a gap of more than 1 hour, we're likely at market close
        let time_diff = next_est.signed_duration_since(est_time);
        if time_diff.num_hours() > 1 {
            return true;
        }
    }
    
    false
}

fn can_open_trade(timestamp: &DateTime<Utc>, bars: &[Bar], current_index: usize) -> bool {
    use chrono::{FixedOffset, Weekday};
    
    // Convert to Eastern Time
    let est_offset = FixedOffset::west_opt(5 * 3600).unwrap();
    let est_time = timestamp.with_timezone(&est_offset);
    
    // Check if it's a weekday
    match est_time.weekday() {
        Weekday::Sat | Weekday::Sun => return false,
        _ => {}
    }
    
    let hour = est_time.hour();
    let minute = est_time.minute();
    
    // Market hours: 9:30 AM - 4:00 PM EST
    // Don't allow new trades after 3:50 PM to ensure they can be closed by 4:00 PM
    let after_open = hour > 9 || (hour == 9 && minute >= 30);
    let before_cutoff = hour < 15 || (hour == 15 && minute <= 50);
    
    if !after_open || !before_cutoff {
        return false;
    }
    
    // Check if market will still be open for the next 2 bars (10 minutes)
    // This ensures we have time to close the position before market close
    for i in 1..=2 {
        if let Some(future_bar) = bars.get(current_index + i) {
            let future_time = future_bar.timestamp.with_timezone(&est_offset);
            // If any of the next 2 bars is at or after 4:00 PM, don't allow trade
            if future_time.hour() >= 16 {
                return false;
            }
            // If it's a different day (market closed overnight), don't allow trade
            if future_time.date_naive() != est_time.date_naive() {
                return false;
            }
        }
    }
    
    true
}

fn generate_sample_bars() -> Vec<Bar> {
    let mut bars = Vec::new();
    let mut price = 1.08500; // EUR/USD starting price
    let now = Utc::now();
    
    // Generate 1000 bars for more history (about 3.5 days of 5-minute bars)
    for i in 0..1000 {
        let timestamp = now - chrono::Duration::minutes(5 * (1000 - i));
        
        // Reduce volatility during non-trading hours
        let volatility_factor = if is_trading_hours(&timestamp) { 1.0 } else { 0.3 };
        
        let change = (rand() - 0.5) * 0.0010 * volatility_factor;
        price += change;
        
        let open = price;
        let close = price + (rand() - 0.5) * 0.0005 * volatility_factor;
        let high = open.max(close) + rand() * 0.0002 * volatility_factor;
        let low = open.min(close) - rand() * 0.0002 * volatility_factor;
        let volume = if is_trading_hours(&timestamp) {
            1000.0 + rand() * 2000.0
        } else {
            100.0 + rand() * 200.0
        };
        
        bars.push(Bar {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        });
        
        price = close;
    }
    
    bars
}

fn rand() -> f64 {
    use std::cell::Cell;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    thread_local! {
        static SEED: Cell<u64> = Cell::new(0);
    }
    
    SEED.with(|seed| {
        let mut hasher = DefaultHasher::new();
        let current = seed.get();
        seed.set(current.wrapping_add(1));
        current.hash(&mut hasher);
        let hash = hasher.finish();
        (hash % 1000) as f64 / 1000.0
    })
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 700.0])
            .with_min_inner_size([200.0, 200.0])  // Allow very small windows
            .with_title("Trading Replay"),
        ..Default::default()
    };
    
    eframe::run_native(
        "Trading Replay",
        options,
        Box::new(|_cc| Ok(Box::new(TradingReplayApp::default()))),
    )
}