import yfinance as yf
import pandas as pd
import numpy as np

# --- MERGED FEATURE ENGINEERING CLASS ---
class FeatureEngineer:
    @staticmethod
    def calculate_bollinger_extension(series: pd.Series, window: int = 20, num_std: int = 2) -> pd.Series:
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        upper_band = rolling_mean + (num_std * rolling_std)
        # Avoid division by zero
        return ((series - upper_band) / (upper_band + 1e-8)) * 100

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        loss = loss.replace(0, np.nan)
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_volatility(series: pd.Series, window: int = 30) -> pd.Series:
        return series.pct_change().rolling(window).std() * (252**0.5) * 100

    @staticmethod
    def calculate_relative_metrics(stock_close: pd.Series, benchmark_close: pd.Series) -> pd.DataFrame:
        if len(stock_close) != len(benchmark_close):
            return pd.DataFrame({'Rel_RSI': np.nan}, index=stock_close.index)
        rs_ratio = stock_close / benchmark_close
        rel_rsi = FeatureEngineer.calculate_rsi(rs_ratio)
        return pd.DataFrame({'Rel_RSI': rel_rsi})

    @staticmethod
    def calculate_relative_volume(volume: pd.Series, window: int = 30) -> pd.Series:
        avg_volume = volume.rolling(window).mean()
        return volume / (avg_volume + 1e-8)

# --- MAIN DATA LOADER ---
class MarketDataLoader:
    _benchmark_cache = {}
    _vix_cache = {}

    def __init__(self, start_date: str = "2018-01-01"):
        self.start_date = start_date
        self.feature_engine = FeatureEngineer()

        if start_date not in MarketDataLoader._benchmark_cache:
            bench_df = yf.Ticker("QQQ").history(start=start_date)
            bench_df.index = pd.to_datetime(bench_df.index).tz_localize(None).normalize()
            bench_df = bench_df[~bench_df.index.duplicated(keep='first')]
            MarketDataLoader._benchmark_cache[start_date] = bench_df
        self.benchmark = MarketDataLoader._benchmark_cache[start_date].copy()

        if start_date not in MarketDataLoader._vix_cache:
            vix_df = yf.Ticker("^VIX").history(start=start_date)
            vix_df.index = pd.to_datetime(vix_df.index).tz_localize(None).normalize()
            vix_df = vix_df[~vix_df.index.duplicated(keep='first')]
            MarketDataLoader._vix_cache[start_date] = vix_df[['Close']].rename(columns={'Close': 'Fear_Gauge'})
        self.vix = MarketDataLoader._vix_cache[start_date].copy()

    def fetch_clean_stock_data(self, ticker: str):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=self.start_date)
            if len(hist) < 250: return None
            
            hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
            hist = hist[~hist.index.duplicated(keep='first')]
            
            # Merges
            hist = hist.join(self.benchmark['Close'].rename('BENCH_Close'), how='inner')
            hist = hist.join(self.vix, how='inner')

            # Features
            hist['BB_Ext'] = self.feature_engine.calculate_bollinger_extension(hist['Close'])
            hist['Run_Up'] = hist['Close'].pct_change(14) * 100
            hist['Volatility'] = self.feature_engine.calculate_volatility(hist['Close'])
            rel_metrics = self.feature_engine.calculate_relative_metrics(hist['Close'], hist['BENCH_Close'])
            hist['Rel_RSI'] = rel_metrics['Rel_RSI']

            earnings = stock.earnings_dates
            if earnings is None or earnings.empty: return None
            
            # Fix column names for yfinance variations
            cols = [c for c in earnings.columns if 'Surprise' in str(c)]
            if not cols: return None
            earnings = earnings.rename(columns={cols[0]: 'Surprise'}).dropna(subset=['Surprise'])
            earnings.index = pd.to_datetime(earnings.index).tz_localize(None).normalize()

            return self._merge_prices_and_events(hist, earnings, ticker)
        except Exception: return None

    def _merge_prices_and_events(self, history, earnings, ticker):
        events = []
        NEUTRAL_THRESH = 0.015 

        for date, row in earnings.iterrows():
            if date not in history.index: continue
            
            idx = history.index.get_loc(date)
            if idx + 5 >= len(history) or idx < 50: continue
            
            price_now = history.iloc[idx]['Close']
            price_future = history.iloc[idx+5]['Close']
            return_pct = (price_future - price_now) / price_now

            target = 2 if return_pct > NEUTRAL_THRESH else (0 if return_pct < -NEUTRAL_THRESH else 1)
            
            rvol = self.feature_engine.calculate_relative_volume(history['Volume']).iloc[idx]
            feats = history.iloc[idx]

            events.append({
                'Ticker': ticker,
                'Date': date,
                'Surprise': row['Surprise'],
                'BB_Ext': feats['BB_Ext'],
                'Run_Up': feats['Run_Up'],
                'Volatility': feats['Volatility'],
                'Rel_RSI': feats['Rel_RSI'],
                'Fear_Gauge': feats['Fear_Gauge'],
                'RVol': rvol,
                'Target': target,
                'Return': return_pct * 100
            })
        return pd.DataFrame(events)