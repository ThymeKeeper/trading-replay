#!/usr/bin/env python3
"""
Analyze market regime changes in SPY data to identify why certain periods are harder to trade
"""

import pandas as pd
import numpy as np
from datetime import datetime
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Load the data
df = pd.read_csv('AMEX_SPY, 5_8ebe6.csv')
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# Calculate various market metrics
def calculate_market_metrics(df):
    """Calculate various metrics that might indicate market regime changes"""
    
    # Basic price movements
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility measures (20-period rolling)
    df['volatility_20'] = df['returns'].rolling(20).std() * np.sqrt(12 * 252)  # Annualized
    df['volatility_60'] = df['returns'].rolling(60).std() * np.sqrt(12 * 252)
    
    # Range metrics
    df['range'] = df['high'] - df['low']
    df['range_pct'] = df['range'] / df['close']
    df['avg_range_20'] = df['range_pct'].rolling(20).mean()
    
    # Trend strength
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_60'] = df['close'].rolling(60).mean()
    df['trend_strength'] = (df['close'] - df['sma_60']) / df['sma_60']
    
    # Mean reversion vs trending
    # Calculate autocorrelation of returns
    df['returns_autocorr'] = df['returns'].rolling(20).apply(lambda x: x.autocorr(lag=1) if len(x) == 20 else np.nan)
    
    # Volume patterns
    if 'volume' in df.columns:
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['relative_volume'] = df['volume'] / df['volume_ma']
    
    # Gap analysis
    df['gap'] = df['open'] - df['close'].shift(1)
    df['gap_pct'] = df['gap'] / df['close'].shift(1)
    
    # Choppiness Index (measures consolidation vs trending)
    def choppiness_index(high, low, close, period=14):
        atr = pd.Series(index=high.index, dtype=float)
        for i in range(1, len(high)):
            tr = max(high.iloc[i] - low.iloc[i], 
                    abs(high.iloc[i] - close.iloc[i-1]), 
                    abs(low.iloc[i] - close.iloc[i-1]))
            atr.iloc[i] = tr
        
        atr_sum = atr.rolling(period).sum()
        high_low_range = high.rolling(period).max() - low.rolling(period).min()
        ci = 100 * np.log10(atr_sum / high_low_range) / np.log10(period)
        return ci
    
    df['choppiness'] = choppiness_index(df['high'], df['low'], df['close'], 20)
    
    # Efficiency Ratio (trending efficiency)
    def efficiency_ratio(close, period=20):
        direction = abs(close - close.shift(period))
        volatility = (abs(close - close.shift(1))).rolling(period).sum()
        er = direction / volatility
        return er
    
    df['efficiency_ratio'] = efficiency_ratio(df['close'], 20)
    
    return df

# Calculate metrics
df = calculate_market_metrics(df)

# Define the periods
profitable_sept_2024 = (df.index >= '2024-09-01') & (df.index < '2024-10-01')
difficult_period = (df.index >= '2024-10-01') & (df.index < '2025-05-01')
profitable_may_2025 = (df.index >= '2025-05-01')

# Analyze differences between periods
def analyze_period(df, mask, period_name):
    """Analyze characteristics of a specific period"""
    period_data = df[mask]
    
    if len(period_data) == 0:
        return None
        
    stats = {
        'period': period_name,
        'start_date': period_data.index[0],
        'end_date': period_data.index[-1],
        'avg_volatility_20': period_data['volatility_20'].mean(),
        'avg_volatility_60': period_data['volatility_60'].mean(),
        'avg_range_pct': period_data['range_pct'].mean(),
        'avg_choppiness': period_data['choppiness'].mean(),
        'avg_efficiency': period_data['efficiency_ratio'].mean(),
        'avg_returns_autocorr': period_data['returns_autocorr'].mean(),
        'trend_changes': ((period_data['trend_strength'] > 0) != (period_data['trend_strength'].shift(1) > 0)).sum(),
        'avg_gap_pct': period_data['gap_pct'].abs().mean(),
        'positive_days_pct': (period_data['returns'] > 0).sum() / len(period_data),
    }
    
    # Add distribution stats
    stats['returns_skew'] = period_data['returns'].skew()
    stats['returns_kurtosis'] = period_data['returns'].kurtosis()
    
    return stats

# Analyze each period
sept_stats = analyze_period(df, profitable_sept_2024, 'September 2024 (Profitable)')
difficult_stats = analyze_period(df, difficult_period, 'Oct 2024 - Apr 2025 (Difficult)')
may_stats = analyze_period(df, profitable_may_2025, 'May 2025+ (Profitable)')

# Print results
print("=== MARKET REGIME ANALYSIS ===\n")

for stats in [sept_stats, difficult_stats, may_stats]:
    if stats:
        print(f"\n{stats['period']}:")
        print(f"  Date Range: {stats['start_date'].date()} to {stats['end_date'].date()}")
        print(f"  Avg Volatility (20-period): {stats['avg_volatility_20']:.2%}")
        print(f"  Avg Range %: {stats['avg_range_pct']:.2%}")
        print(f"  Choppiness Index: {stats['avg_choppiness']:.2f} {'(Choppy)' if stats['avg_choppiness'] > 61.8 else '(Trending)'}")
        print(f"  Efficiency Ratio: {stats['avg_efficiency']:.3f} {'(Efficient trend)' if stats['avg_efficiency'] > 0.3 else '(Inefficient)'}")
        print(f"  Returns Autocorrelation: {stats['avg_returns_autocorr']:.3f} {'(Mean reverting)' if stats['avg_returns_autocorr'] < -0.1 else '(Trending)' if stats['avg_returns_autocorr'] > 0.1 else '(Random)'}")
        print(f"  Trend Changes: {stats['trend_changes']}")
        print(f"  Avg Gap %: {stats['avg_gap_pct']:.3%}")
        print(f"  Positive Periods: {stats['positive_days_pct']:.1%}")
        print(f"  Returns Skew: {stats['returns_skew']:.3f}")
        print(f"  Returns Kurtosis: {stats['returns_kurtosis']:.3f}")

# Identify key differences
print("\n\n=== KEY DIFFERENCES ===")
if difficult_stats and (sept_stats or may_stats):
    good_stats = sept_stats or may_stats
    
    print("\nDifficult Period Characteristics:")
    if difficult_stats['avg_choppiness'] > good_stats['avg_choppiness'] * 1.1:
        print(f"  - {(difficult_stats['avg_choppiness'] / good_stats['avg_choppiness'] - 1) * 100:.1f}% MORE CHOPPY (consolidating)")
    
    if difficult_stats['avg_efficiency'] < good_stats['avg_efficiency'] * 0.9:
        print(f"  - {(1 - difficult_stats['avg_efficiency'] / good_stats['avg_efficiency']) * 100:.1f}% LESS EFFICIENT trends")
    
    if abs(difficult_stats['avg_returns_autocorr']) < abs(good_stats['avg_returns_autocorr']) * 0.5:
        print(f"  - More RANDOM price action (less predictable)")
    
    if difficult_stats['avg_volatility_20'] != good_stats['avg_volatility_20']:
        vol_diff = (difficult_stats['avg_volatility_20'] / good_stats['avg_volatility_20'] - 1) * 100
        print(f"  - {abs(vol_diff):.1f}% {'HIGHER' if vol_diff > 0 else 'LOWER'} volatility")

# Create visualization if matplotlib is available
if MATPLOTLIB_AVAILABLE:
    plt.figure(figsize=(15, 10))

    # Plot 1: Price with regime periods
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['close'], 'k-', alpha=0.5, linewidth=0.5)
    plt.fill_between(df.index, df['close'].min(), df['close'].max(), 
                     where=profitable_sept_2024, alpha=0.3, color='green', label='Profitable (Sept)')
    plt.fill_between(df.index, df['close'].min(), df['close'].max(), 
                     where=difficult_period, alpha=0.3, color='red', label='Difficult')
    plt.fill_between(df.index, df['close'].min(), df['close'].max(), 
                     where=profitable_may_2025, alpha=0.3, color='green', label='Profitable (May+)')
    plt.title('SPY Price with Trading Period Performance')
    plt.legend()
    plt.ylabel('Price')

    # Plot 2: Choppiness Index
    plt.subplot(3, 1, 2)
    plt.plot(df.index, df['choppiness'], 'b-', alpha=0.7)
    plt.axhline(y=61.8, color='r', linestyle='--', label='Choppy threshold')
    plt.axhline(y=38.2, color='g', linestyle='--', label='Trending threshold')
    plt.fill_between(df.index, 0, 100, where=difficult_period, alpha=0.2, color='red')
    plt.title('Choppiness Index (Higher = More Consolidation)')
    plt.legend()
    plt.ylabel('Choppiness')
    plt.ylim(30, 70)

    # Plot 3: Efficiency Ratio
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['efficiency_ratio'], 'g-', alpha=0.7)
    plt.fill_between(df.index, 0, 1, where=difficult_period, alpha=0.2, color='red')
    plt.axhline(y=0.3, color='b', linestyle='--', label='Efficient trend threshold')
    plt.title('Efficiency Ratio (Higher = Stronger Trends)')
    plt.legend()
    plt.ylabel('Efficiency')
    plt.ylim(0, 0.6)

    plt.tight_layout()
    plt.savefig('market_regime_analysis.png', dpi=150)
    print("\n\nVisualization saved as 'market_regime_analysis.png'")
else:
    print("\n\nMatplotlib not available - skipping visualization")

# Suggest regime detection thresholds
print("\n\n=== SUGGESTED REGIME DETECTION ===")
print("\nBased on the analysis, consider switching strategies when:")
print("1. Choppiness Index > 60: Market is consolidating (range-bound)")
print("2. Efficiency Ratio < 0.25: Trends are weak and unreliable")
print("3. Rolling 20-bar returns autocorrelation near 0: Price action is random")
print("4. Volatility significantly different from baseline")

# Calculate current regime
if len(df) > 0:
    current = df.iloc[-1]
    print(f"\nCurrent Market State:")
    print(f"  Choppiness: {current['choppiness']:.1f} {'(Choppy)' if current['choppiness'] > 61.8 else '(Trending)'}")
    print(f"  Efficiency: {current['efficiency_ratio']:.3f} {'(Efficient)' if current['efficiency_ratio'] > 0.3 else '(Inefficient)'}")
    print(f"  Volatility: {current['volatility_20']:.1%}")