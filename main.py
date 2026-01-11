"""
main.py
-------
FINAL VERSION: NASDAQ-100 3-CLASS STRATEGY.
- Universe: Nasdaq-100 (Live from Wikipedia).
- Target: 3-Class (0=Down, 1=Neutral, 2=Up).
- Strategy: Active Long/Short vs Passive QQQ.
- Models: LogReg, RandomForest, XGBoost, NeuralNet, Ensemble.
"""
import sys
import os
import requests
from io import StringIO

if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import pandas as pd
import numpy as np
import threading
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.data_loader import MarketDataLoader
from src.models import ModelOrchestrator
from src.evaluation import Visualizer, print_professional_dashboard
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# ==========================================
# 1. DYNAMIC CONFIGURATION (NASDAQ-100)
# ==========================================

def fetch_live_nasdaq_100():
    print("# Connecting to Wikipedia to fetch live Nasdaq-100 list...")
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        r = requests.get(url, headers=headers)
        r.raise_for_status()

        tables = pd.read_html(StringIO(r.text))

        target_table = None
        for t in tables:
            if 'Ticker' in t.columns: target_table = t; break
            elif 'Symbol' in t.columns: target_table = t; break

        if target_table is None: raise ValueError("Constituents table not found.")

        col = 'Ticker' if 'Ticker' in target_table.columns else 'Symbol'
        tickers = target_table[col].astype(str).tolist()
        cleaned_tickers = [t.replace('.', '-') for t in tickers]

        print(f"# [OK] Success: Retrieved {len(cleaned_tickers)} active tickers.")
        return cleaned_tickers

    except Exception as e:
        print(f"\n[ERROR] CRITICAL ERROR: {e}")
        print("   Using fallback list...")
        return ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'AVGO', 'PEP', 'COST', 'CSCO', 'TMUS']

# --- CONFIGURATION ---
TICKERS = fetch_live_nasdaq_100()
START_DATE = "2019-01-01"

FEATURES = [
    'Surprise', 'Fear_Gauge', 'Run_Up', 'BB_Ext',
    'Volatility', 'Priced_In', 'Rel_RSI',
    'RVol'
]

# Updated to use new structure
CACHE_DIR = "data/raw"
CACHE_FILE = os.path.join(CACHE_DIR, "cache_nasdaq_3class_final.csv")
RESULTS_DIR = "results"

MODELS = ['Logistic Regression', 'Random Forest', 'Neural Network', 'XGBoost', 'Ensemble']

def calculate_priced_in_feature(df):
    s_mean = df['Surprise'].rolling(20, min_periods=1).mean()
    s_std = df['Surprise'].rolling(20, min_periods=1).std()
    r_mean = df['Run_Up'].rolling(20, min_periods=1).mean()
    r_std = df['Run_Up'].rolling(20, min_periods=1).std()
    epsilon = 1e-8
    norm_surprise = (df['Surprise'] - s_mean) / (s_std + epsilon)
    norm_runup = (df['Run_Up'] - r_mean) / (r_std + epsilon)
    df['Priced_In'] = (norm_surprise.fillna(0) * norm_runup.fillna(0))
    return df

def check_correlations(df, features, threshold=0.90):
    print("\n[ANALYZING] Analyzing Feature Correlations...")
    corr_matrix = df[features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    if to_drop:
        print(f"[WARNING] Removing redundant features (> {threshold}): {to_drop}")
        return [f for f in features if f not in to_drop]
    return features

def main():
    print("\n" + "=" * 70)
    print("[NASDAQ-100] NASDAQ-100 3-CLASS PREDICTOR (FINAL RUN)")
    print(f"   [Strategy: Up/Neutral/Down] [Benchmark: QQQ]")
    print("=" * 70)

    # Create necessary directories
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    loader = MarketDataLoader(start_date=START_DATE)
    orchestrator = ModelOrchestrator()
    viz = Visualizer(output_dir=RESULTS_DIR)

    # ---------------------------------------------------------
    # PHASE 1: DATA ACQUISITION
    # ---------------------------------------------------------
    master_df = None
    if os.path.exists(CACHE_FILE):
        try:
            master_df = pd.read_csv(CACHE_FILE)
            master_df['Date'] = pd.to_datetime(master_df['Date'])
            if 'Fear_Gauge' in master_df.columns:
                print(f"  [OK] Cache loaded: {len(master_df):,} events.")
        except: master_df = None

    if master_df is None:
        all_datasets = []
        print(f"  ⬇️  Downloading historical data for {len(TICKERS)} tickers...")

        loader_local = threading.local()
        def download_ticker(ticker):
            if not hasattr(loader_local, "loader"):
                loader_local.loader = MarketDataLoader(start_date=START_DATE)
            return loader_local.loader.fetch_clean_stock_data(ticker)

        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_ticker = {executor.submit(download_ticker, ticker): ticker for ticker in TICKERS}

            with tqdm(total=len(TICKERS), desc="📊 Downloading NASDAQ-100 Data",
                     unit="ticker", ncols=100, bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

                for future in as_completed(future_to_ticker):
                    df = future.result()
                    if df is not None:
                        all_datasets.append(df)
                    pbar.update(1)

        if not all_datasets: sys.exit("[ERROR] Error: No data collected.")
        master_df = pd.concat(all_datasets, ignore_index=True).sort_values('Date').reset_index(drop=True)
        master_df = calculate_priced_in_feature(master_df)
        master_df.to_csv(CACHE_FILE, index=False)

    viz.plot_correlation_heatmap(master_df, FEATURES)
    FINAL_FEATURES = check_correlations(master_df, FEATURES)
    print(f"   → Final Features ({len(FINAL_FEATURES)}): {FINAL_FEATURES}")

    # ---------------------------------------------------------
    # PHASE 2: EXPANDING WINDOW ANALYSIS (Yearly)
    # ---------------------------------------------------------
    print("\n┌" + "─" * 68 + "┐")
    print("│ PHASE 2: EXPANDING WINDOW ANALYSIS (Yearly)" + " " * 29 + "│")
    print("└" + "─" * 68 + "┘")

    results_log = []
    master_df = master_df.sort_values('Date').reset_index(drop=True)
    end_year = master_df['Date'].dt.year.max()

    print("  📊 Expanding Window Validation Schedule:")
    print("  ────────────────────────────────────────")

    for test_year in range(2023, end_year + 1):
        train_end_year = test_year - 1
        train_mask = master_df['Date'].dt.year <= train_end_year
        train_data = master_df[train_mask]
        test_mask = master_df['Date'].dt.year == test_year
        test_data = master_df[test_mask]

        if len(train_data) < 200:
            print(f"  ⚠️  Skipping {test_year}: Insufficient training data ({len(train_data)} events)")
            continue

        if len(test_data) == 0:
            print(f"  ⚠️  Skipping {test_year}: No test data available")
            continue

        print(f"  🔄 Train: 2019-{train_end_year} ({len(train_data):>4} events) → Test: {test_year} ({len(test_data):>3} events)", end=" ... ", flush=True)

        train_weights = train_data['Return'].abs().values

        # Tune hyperparameters once per year (not per event)
        orchestrator.tune_hyperparameters(train_data[FINAL_FEATURES], train_data['Target'], sample_weight=train_weights)

        # Train models once per year and predict on all test events at once
        X_train = train_data[FINAL_FEATURES]
        y_train = train_data['Target']
        X_test = test_data[FINAL_FEATURES]

        preds, _ = orchestrator.train_and_evaluate(X_train, y_train, X_test, train_weights)

        # Add predictions to results log
        for idx, (_, test_event) in enumerate(test_data.iterrows()):
            result_row = {
                'Date': test_event['Date'],
                'Ticker': test_event['Ticker'],
                'Target': test_event['Target'],
                'Actual_Return': test_event['Return']
            }
            # Add predictions from each model
            for model_name in preds:
                result_row[model_name] = preds[model_name][idx]

            results_log.append(result_row)

        print(f"✓ ({len(test_data)} predictions)")

    print(f"\n  ✅ Completed expanding window validation with {len(results_log)} total predictions\n")

    # ---------------------------------------------------------
    # PHASE 3: PERFORMANCE ANALYSIS
    # ---------------------------------------------------------
    print("\n┌" + "─" * 68 + "┐")
    print("│ PHASE 3: PERFORMANCE ANALYSIS" + " " * 38 + "│")
    print("└" + "─" * 68 + "┘")

    results_df = pd.DataFrame(results_log)
    initial_capital = 10000
    portfolio_df = results_df.copy()

    def map_signal(x):
        if x == 2: return 1
        if x == 0: return -1
        return 0

    portfolio_df['Position'] = portfolio_df['Ensemble'].apply(map_signal)
    max_position_size = 0.02
    portfolio_df['Size'] = max_position_size
    STOP_LOSS_PCT = -10.0
    portfolio_df['Actual_Return_SL'] = portfolio_df['Actual_Return'].clip(lower=STOP_LOSS_PCT)
    portfolio_df['Trade_PnL'] = portfolio_df['Position'] * portfolio_df['Size'] * (portfolio_df['Actual_Return_SL'] / 100)
    portfolio_df['Equity_Strategy'] = initial_capital + (initial_capital * portfolio_df['Trade_PnL']).cumsum()

    # Benchmark
    print("  → Calculating Passive QQQ Benchmark...")
    try:
        min_date = portfolio_df['Date'].min()
        max_date = portfolio_df['Date'].max()
        qqq_hist = yf.Ticker("QQQ").history(start=min_date - pd.Timedelta(days=7), end=max_date + pd.Timedelta(days=7))

        if qqq_hist.empty:
            print("  [WARNING] Warning: Could not fetch QQQ data, using flat benchmark")
            benchmark_equity = [initial_capital] * len(portfolio_df)
        else:
            qqq_hist.index = qqq_hist.index.tz_localize(None)
            start_price = qqq_hist['Close'].iloc[0]
            print(f"  → QQQ Start Price: ${start_price:.2f} on {qqq_hist.index[0].date()}")

            benchmark_equity = []
            for d in portfolio_df['Date']:
                if isinstance(d, str): d = pd.to_datetime(d)
                if hasattr(d, 'tz') and d.tz is not None: d = d.tz_localize(None)
                idx = qqq_hist.index.get_indexer([d], method='nearest')[0]
                current_price = qqq_hist['Close'].iloc[idx]
                benchmark_equity.append((current_price / start_price) * initial_capital)

            end_price = qqq_hist['Close'].iloc[-1]
            print(f"  → QQQ End Price: ${end_price:.2f} on {qqq_hist.index[-1].date()}")
    except Exception as e:
        print(f"  [WARNING] Warning: Benchmark calculation error: {e}")
        benchmark_equity = [initial_capital] * len(portfolio_df)

    portfolio_df['Equity_Benchmark'] = benchmark_equity

    strat_final = portfolio_df['Equity_Strategy'].iloc[-1]
    bench_final = portfolio_df['Equity_Benchmark'].iloc[-1]
    strat_total_ret = ((strat_final/initial_capital)-1)*100
    bench_total_ret = ((bench_final/initial_capital)-1)*100
    outperformance = strat_total_ret - bench_total_ret

    results_df['Ensemble_Equity'] = portfolio_df['Equity_Strategy'] - initial_capital
    results_df['QQQ_Benchmark_Equity'] = portfolio_df['Equity_Benchmark'] - initial_capital

    n_long = (portfolio_df['Position'] == 1).sum()
    n_short = (portfolio_df['Position'] == -1).sum()
    n_neutral = (portfolio_df['Position'] == 0).sum()
    n_stopped = (portfolio_df['Actual_Return'] < STOP_LOSS_PCT).sum()
    total_trades = n_long + n_short
    stop_rate = (n_stopped / total_trades * 100) if total_trades > 0 else 0
    losses_without_sl = portfolio_df[portfolio_df['Actual_Return'] < STOP_LOSS_PCT]['Actual_Return']
    loss_prevented = (losses_without_sl - STOP_LOSS_PCT).sum() if len(losses_without_sl) > 0 else 0

    print("\n┌" + "─" * 68 + "┐")
    print("│ PORTFOLIO PERFORMANCE" + " " * 46 + "│")
    print("└" + "─" * 68 + "┘")
    print(f"  Strategy Equity:    ${strat_final:>12,.2f}")
    print(f"  Benchmark (QQQ):    ${bench_final:>12,.2f}")
    print("-" * 70)
    print(f"  Strategy Return:    {strat_total_ret:>12.2f}%")
    print(f"  Benchmark Return:   {bench_total_ret:>12.2f}%")
    print(f"  Outperformance:     {outperformance:>12.2f}%")
    print("-" * 70)
    print(f"  Trades: {n_long+n_short} (L: {n_long} | S: {n_short} | Neut: {n_neutral})")
    print(f"  Stop Loss Hits:     {n_stopped:>3} trades ({stop_rate:>5.1f}%)")
    print(f"  Loss Prevented:     {loss_prevented:>12.2f}%")
    print("─" * 70)

    # Metrics
    metrics_list = []
    active_models = [m for m in MODELS if m in results_df.columns]

    for m in active_models:
        acc = accuracy_score(results_df['Target'], results_df[m])
        prec = precision_score(results_df['Target'], results_df[m], average='weighted', zero_division=0)
        rec = recall_score(results_df['Target'], results_df[m], average='weighted', zero_division=0)
        f1 = f1_score(results_df['Target'], results_df[m], average='weighted', zero_division=0)

        try:
            y_true_bin = label_binarize(results_df['Target'], classes=[0, 1, 2])
            y_pred_bin = label_binarize(results_df[m], classes=[0, 1, 2])
            roc_auc = roc_auc_score(y_true_bin, y_pred_bin, average='weighted', multi_class='ovr')
        except:
            roc_auc = 0.0

        metrics_list.append({
            'Model': m,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC AUC': roc_auc
        })
    metrics_df = pd.DataFrame(metrics_list).set_index('Model')

    print("\n┌" + "─" * 100 + "┐")
    print("│ MODEL PERFORMANCE SCOREBOARD" + " " * 71 + "│")
    print("└" + "─" * 100 + "┘")
    print(f"{'Model':<25} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'ROC AUC':>12}")
    print("─" * 102)
    for idx, row in metrics_df.iterrows():
        print(f"{idx:<25} {row['Accuracy']:>12.4f} {row['Precision']:>12.4f} {row['Recall']:>12.4f} {row['F1-Score']:>12.4f} {row['ROC AUC']:>12.4f}")
    print("─" * 102)

    best_acc = metrics_df['Accuracy'].idxmax()
    best_prec = metrics_df['Precision'].idxmax()
    best_rec = metrics_df['Recall'].idxmax()
    best_f1 = metrics_df['F1-Score'].idxmax()
    best_roc = metrics_df['ROC AUC'].idxmax()

    print("\n[BEST PERFORMERS]:")
    print(f"   • Accuracy:  {best_acc} ({metrics_df.loc[best_acc, 'Accuracy']:.4f})")
    print(f"   • Precision: {best_prec} ({metrics_df.loc[best_prec, 'Precision']:.4f})")
    print(f"   • Recall:    {best_rec} ({metrics_df.loc[best_rec, 'Recall']:.4f})")
    print(f"   • F1-Score:  {best_f1} ({metrics_df.loc[best_f1, 'F1-Score']:.4f})")
    print(f"   • ROC AUC:   {best_roc} ({metrics_df.loc[best_roc, 'ROC AUC']:.4f})")
    print("─" * 102 + "\n")

    if 'Random Forest' in orchestrator.models:
        rf_model = orchestrator.models['Random Forest']
        X_full = master_df[FINAL_FEATURES]
        y_full = master_df['Target']
        rf_model.fit(X_full, y_full)
        viz.plot_feature_importance(rf_model, FINAL_FEATURES)
        viz.plot_shap_analysis(rf_model, X_full, FINAL_FEATURES)
        print_professional_dashboard(results_df, metrics_df, portfolio_df, rf_model, FINAL_FEATURES)
    else:
        print_professional_dashboard(results_df, metrics_df, portfolio_df, None, None)

    viz.plot_equity_curve(results_df, ['Ensemble', 'QQQ_Benchmark'])
    viz.plot_model_comparison(metrics_df)
    viz.plot_trade_performance_distribution(portfolio_df)

    portfolio_df.to_csv(os.path.join(RESULTS_DIR, "final_results_3class.csv"), index=False)
    print("\n[SUCCESS] Analysis Complete.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED] Execution interrupted by user.")
        sys.exit(0)
