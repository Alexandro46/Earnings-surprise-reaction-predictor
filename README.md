# Earnings Surprise Reaction Predictor

A machine learning system that predicts stock price reactions to earnings announcements using technical indicators and earnings surprises for NASDAQ-100 stocks.

## Features

### 🎯 Core Capabilities
- **Expanding Window Validation**: Realistic backtesting with no look-ahead bias
- **Ensemble Models**: Combines Logistic Regression, Random Forest, Neural Networks, and XGBoost
- **Advanced Features**: 8 engineered features including relative strength, volatility, and market fear gauge
- **Portfolio Simulation**: Long/Short strategy with realistic position sizing and stop-loss protection
- **3-Class Prediction**: Up (2), Neutral (1), or Down (0) classifications

### 📊 Technical Features
1. **Earnings Surprise** - EPS beat/miss magnitude
2. **Bollinger Band Extension** - Price deviation from bands
3. **Run-Up** - 14-day pre-earnings price movement
4. **Volatility** - Annualized historical volatility
5. **Priced-In** - Interaction between surprise and run-up
6. **Relative RSI** - Stock RSI vs benchmark
7. **Fear_Gauge** - VIX volatility index level
8. **Relative Volume (RVol)** - Trading volume vs 30-day average

---

## Quick Start

### Installation

#### Option 1: Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install core dependencies (~5 minutes, ~500MB)
pip install -r requirements.txt

# Run the project
python main.py
```

#### Option 2: System-wide Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the project
python main.py
```

### Basic Usage
```bash
# Run the predictor
python main.py
```

**First run**: Downloads NASDAQ-100 stocks from Wikipedia (5-10 minutes)
**Subsequent runs**: Uses cache (instant)

---

## Project Structure

```
Earnings-surprise-reaction-predictor-main/
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── data/
│   └── raw/
│       └── cache_nasdaq_3class_final.csv    # Cached market data
├── results/
│   ├── final_results_3class.csv             # Portfolio results
│   ├── equity_curve.png                     # Portfolio performance chart
│   ├── correlation_heatmap.png              # Feature correlation matrix
│   ├── feature_importance.png               # Random Forest feature weights
│   ├── shap_summary.png                     # SHAP analysis (optional)
│   ├── model_comparison.png                 # Model metrics comparison
│   └── trade_performance_distribution.png   # Trade returns histogram
└── src/
    ├── __init__.py             # Package initialization
    ├── data_loader.py          # Yahoo Finance API + feature engineering
    ├── models.py               # ML models (LogReg, RF, XGBoost, NN, Ensemble)
    └── evaluation.py           # Charts and terminal dashboard
```

---

## Configuration

### Ticker Universe
The system automatically fetches the live NASDAQ-100 constituent list from Wikipedia.

**Default**: NASDAQ-100 (~100 stocks)
**Fallback**: If Wikipedia fetch fails, uses a curated list of major tech stocks

To customize the ticker list, edit the `fetch_live_nasdaq_100()` function in [main.py](main.py#L34-63).

### Date Range
```python
START_DATE = "2019-01-01"  # Line 67 in main.py
```
**Current setting**: 2019-2025 (~6 years)
- Includes COVID period and post-pandemic recovery
- More recent market conditions

**Alternative**: Change to "2015-01-01" for 10+ years of data

### Model Selection
```python
MODELS = ['Logistic Regression', 'Random Forest', 'Neural Network', 'XGBoost', 'Ensemble']
```
All models are trained and evaluated. The Ensemble model uses majority voting across all individual models.

### Features Configuration
Edit the `FEATURES` list in [main.py](main.py#L69-73) to customize which features are used for training.

---

## Output Files

### Generated Charts

| File | Description |
|------|-------------|
| `results/equity_curve.png` | Portfolio value over time with drawdown analysis |
| `results/trade_performance_distribution.png` | Histogram of trade returns by range |
| `results/feature_importance.png` | Random Forest feature importance |
| `results/shap_summary.png` | SHAP feature impact analysis (if SHAP installed) |
| `results/model_comparison.png` | Model performance metrics comparison |
| `results/correlation_heatmap.png` | Feature correlation analysis |

### Data Files

| File | Description |
|------|-------------|
| `data/raw/cache_nasdaq_3class_final.csv` | Cached earnings events with features |
| `results/final_results_3class.csv` | All predictions, actual returns, and equity curves |

---

## Methodology

### Expanding Window Analysis
The system uses expanding window validation to prevent look-ahead bias:
- **Train**: 2019-2022 → **Test**: 2023
- **Train**: 2019-2023 → **Test**: 2024
- **Train**: 2019-2024 → **Test**: 2025

Each year's test data is never seen during training, ensuring realistic backtesting.

### Position Sizing & Risk Management
- **Position Size**: 2% of portfolio per trade
- **Stop Loss**: -10% per trade to limit downside
- **Strategy**:
  - Class 2 (Up) → Long position (+1)
  - Class 0 (Down) → Short position (-1)
  - Class 1 (Neutral) → No position (0)

### Performance Metrics
- **Accuracy**: % of correct predictions (3-class)
- **Precision/Recall/F1**: Balanced performance metrics
- **ROC AUC**: Multi-class area under curve
- **Strategy Return**: Total portfolio performance
- **Outperformance**: Strategy return vs QQQ buy-and-hold benchmark

### Feature Engineering
Features are calculated per earnings event:
- **Technical indicators**: BB Extension, RSI, Volatility
- **Relative metrics**: Stock vs QQQ benchmark performance
- **Volume analysis**: Relative volume vs 30-day average
- **Interaction features**: Priced-In (surprise × run-up)
- **Market regime**: VIX fear gauge at time of earnings

---

## Data Sources

- ✅ **Yahoo Finance** - Stock prices, earnings dates, EPS surprises
- ✅ **Technical Indicators** - Calculated from price/volume data
- ✅ **VIX Data** - CBOE Volatility Index from Yahoo Finance
- ✅ **QQQ Benchmark** - NASDAQ-100 ETF for relative performance
- ✅ **Wikipedia** - Live NASDAQ-100 constituent list

---

## Performance Expectations

### Realistic Goals (6 years of data)
- **Accuracy**: 40-55% (3-class classification)
- **F1-Score**: 0.40-0.55
- **Outperformance**: -5% to +15% vs QQQ benchmark

### Red Flags (Possible Issues)
- Accuracy >75%: Likely data leakage
- Outperformance >25%: Check for bugs
- All models agree 100%: Overfitting

### Good Results
- Consistent accuracy across models
- Stable performance over time windows
- Logical feature importance (e.g., Surprise, Run_Up ranked high)
- Positive Sharpe ratio with reasonable drawdowns

---

## Troubleshooting

### "Insufficient data" error
```bash
pip install --upgrade yfinance
# Delete data/raw/cache_nasdaq_3class_final.csv and re-run
rm -rf data/raw/cache_nasdaq_3class_final.csv  # macOS/Linux
del data\raw\cache_nasdaq_3class_final.csv     # Windows
python main.py
```

### XGBoost not found
```bash
pip install xgboost
# Or continue without it - system will work with 4 models instead of 5
```

### SHAP not found
```bash
pip install shap
# Or continue without it - you'll get basic feature importance instead
```

### Slow data download
- Normal on first run (5-10 minutes for ~100 stocks)
- Uses cache after first successful run
- Reduce ticker list to speed up initial download

### Wikipedia fetch fails
The system has a fallback list of major NASDAQ-100 stocks. Check your internet connection or wait a few minutes if Wikipedia is temporarily unavailable.

---

## Scientific Validation

This project follows industry best practices:
- ✅ Expanding window validation (no lookahead bias)
- ✅ Multiple models for robustness
- ✅ Proper train/test split by time
- ✅ 3-class classification for nuanced predictions
- ✅ Out-of-sample testing only
- ✅ Sample weighting by return volatility
- ✅ Hyperparameter tuning on training data only
- ✅ Stop-loss protection for risk management

---

## Future Improvements

### Low-Hanging Fruit
1. Tune the neutral threshold (currently ±1.5%)
2. Test different lookback periods for technical indicators
3. Experiment with ensemble weights instead of simple voting
4. Add more sector-specific features

### Advanced
1. Add sentiment analysis from earnings call transcripts
2. Incorporate analyst estimate consensus data
3. Multi-timeframe features (daily + intraday)
4. Options market data (IV skew, put/call ratio)
5. News sentiment before/after earnings

---

## Technical Details

### Data Collection
- Earnings dates and surprises from Yahoo Finance
- 250+ days of historical data required per stock
- Automatic handling of stock splits and dividends
- Timezone-aware date processing

### Feature Engineering
- Rolling windows for technical indicators
- Benchmark-relative calculations
- NaN handling with forward fill
- Standardization for neural networks

### Model Training
- Class balancing for imbalanced targets
- Sample weighting by return magnitude
- RandomizedSearchCV for hyperparameter tuning
- Ensemble via majority voting

---

## License

Educational/Research use. Not financial advice.

---

## Questions?

1. Check code comments for implementation details
2. Review `results/final_results_3class.csv` for detailed results
3. Examine terminal output for performance metrics

**Remember**: This is a research tool, not a trading system. Always validate results and understand the limitations before making any real-world decisions.
