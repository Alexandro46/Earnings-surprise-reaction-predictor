"""
src/evaluation.py
-----------------
Unified module for metrics, plots, and visualization.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not found. Install with 'pip install shap' for advanced feature importance analysis.")


def print_professional_dashboard(results_df, metrics_df, portfolio_df, rf_model=None, feature_names=None):
    """Prints a concise terminal dashboard."""
    print("\n" + "="*80)
    print("📊 NASDAQ-100 STRATEGY REPORT - FINAL")
    print("="*80)

    n_events = len(results_df)
    date_start = str(results_df['Date'].min())[:10] if not results_df.empty else "N/A"
    date_end = str(results_df['Date'].max())[:10] if not results_df.empty else "N/A"

    print("[SYSTEM STATUS]")
    print(f"• Universe:      {results_df['Ticker'].nunique()} Stocks")
    print(f"• Date Range:    {date_start} to {date_end}")
    print(f"• Events Mined:  {n_events} Earnings Calls")

    # Feature Importance
    if rf_model is not None and feature_names is not None:
        print("\n" + "-"*80)
        print("🔍 CRITERIA WEIGHT & IMPACT ANALYSIS")
        print("-" * 80)
        print(f"{'Criteria (Feature)':<25} | {'Weight %':<10} | {'Influence'}")
        print("-" * 80)
        try:
            importances = rf_model.feature_importances_
            feature_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feature_imp = feature_imp.sort_values('Importance', ascending=False)
            for _, row in feature_imp.iterrows():
                weight = row['Importance'] * 100
                desc = "🔥 MAJOR" if weight > 15 else "🔸 MODERATE" if weight > 5 else "▫️ LOW"
                print(f"{row['Feature']:<25} | {weight:>6.1f}%    | {desc}")
        except: pass


class Visualizer:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        os.makedirs(output_dir, exist_ok=True)

    def plot_correlation_heatmap(self, df: pd.DataFrame, features: list):
        """Plots correlation matrix to detect multicollinearity."""
        plt.figure(figsize=(10, 8))
        corr = df[features].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                    vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
        plt.title("Feature Correlation Matrix (Check for Multicollinearity)", fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/correlation_heatmap.png", dpi=300)
        plt.close()
        print(f"✓ Chart saved: {self.output_dir}/correlation_heatmap.png\n")

    def plot_equity_curve(self, results: pd.DataFrame, models: list):
        """Enhanced equity curve plot with drawdown analysis."""
        available_models = [m for m in models if f'{m}_Equity' in results.columns]
        if not available_models: return

        final_scores = {m: results[f'{m}_Equity'].iloc[-1] for m in available_models}

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1], sharex=True)

        # Main equity curve
        for model in available_models:
            equity_series = results[f'{model}_Equity']
            color = 'gray' if "Benchmark" in model else '#2ecc71'
            style = '--' if "Benchmark" in model else '-'
            alpha = 0.7 if "Benchmark" in model else 1.0
            width = 2 if "Benchmark" in model else 2.5

            label_txt = f"{model.replace('_Equity','')} (${final_scores[model]:+,.2f})"
            ax1.plot(results['Date'], equity_series,
                     label=label_txt, color=color, linestyle=style, linewidth=width, alpha=alpha)

            if "Benchmark" not in model:
                ax1.fill_between(results['Date'], equity_series, 0,
                                where=(equity_series >= 0), color=color, alpha=0.1)
                ax1.fill_between(results['Date'], equity_series, 0,
                                where=(equity_series < 0), color='red', alpha=0.1)

        ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax1.set_title("Portfolio Performance Over Time", fontweight='bold', fontsize=14)
        ax1.set_ylabel("Portfolio Value ($)", fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Drawdown plot
        for model in available_models:
            equity_series = results[f'{model}_Equity']
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / (peak + 1e-8) * 100

            color = 'gray' if "Benchmark" in model else '#e74c3c'
            style = '--' if "Benchmark" in model else '-'
            alpha = 0.7 if "Benchmark" in model else 1.0

            label_txt = f"{model.replace('_Equity','')} Drawdown"
            ax2.plot(results['Date'], drawdown,
                     label=label_txt, color=color, linestyle=style, linewidth=1.5, alpha=alpha)
            ax2.fill_between(results['Date'], drawdown, 0, color=color, alpha=0.2)

        ax2.set_title("Portfolio Drawdown", fontweight='bold', fontsize=12)
        ax2.set_ylabel("Drawdown (%)", fontweight='bold')
        ax2.set_xlabel("Date", fontweight='bold')
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        # Add performance summary
        if available_models:
            strategy_model = [m for m in available_models if "Benchmark" not in m][0]
            bench_model = [m for m in available_models if "Benchmark" in m][0] if any("Benchmark" in m for m in available_models) else None

            strat_final = final_scores[strategy_model]
            strat_return = (strat_final / 10000 - 1) * 100
            summary_text = f"Strategy: ${strat_final:,.0f} ({strat_return:+.1f}%)"

            if bench_model:
                bench_final = final_scores[bench_model]
                bench_return = (bench_final / 10000 - 1) * 100
                outperformance = strat_return - bench_return
                summary_text += f"\nBenchmark: ${bench_final:,.0f} ({bench_return:+.1f}%)"
                summary_text += f"\nOutperformance: {outperformance:+.1f}%"

            strategy_equity = results[f'{strategy_model}_Equity']
            peak = strategy_equity.expanding().max()
            max_drawdown = ((strategy_equity - peak) / (peak + 1e-8)).min() * 100
            summary_text += f"\nMax Drawdown: {max_drawdown:.1f}%"

            ax1.text(0.02, 0.98, summary_text, transform=ax1.transAxes,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/equity_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Enhanced equity curve with drawdown saved: {self.output_dir}/equity_curve.png")

    def plot_feature_importance(self, model, feature_names: list):
        if not hasattr(model, 'feature_importances_'): return
        importances = model.feature_importances_
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances, y=feature_names, palette="viridis")
        plt.title("Feature Importance", fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Chart saved: {self.output_dir}/feature_importance.png")

    def plot_shap_analysis(self, model, X_train: pd.DataFrame, feature_names: list, max_evals: int = 100):
        """Generate SHAP analysis plots for model interpretability."""
        if not SHAP_AVAILABLE:
            print("⚠️ SHAP not available. Skipping SHAP analysis.")
            return

        try:
            print("🔍 Computing SHAP values for model interpretability...")
            sample_size = min(1000, len(X_train))
            X_sample = X_train.sample(n=sample_size, random_state=42)

            # Use TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_sample)

            # SHAP Summary Plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            plt.title("SHAP Feature Impact Summary", fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/shap_summary.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ SHAP Summary Plot saved: {self.output_dir}/shap_summary.png")

        except Exception as e:
            print(f"⚠️ SHAP analysis failed: {e}")

    def plot_model_comparison(self, perf_df: pd.DataFrame):
        """Compare model performance across metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        metrics = ['Accuracy', 'ROC AUC', 'F1-Score']
        colors = sns.color_palette("husl", len(perf_df))

        for idx, metric in enumerate(metrics):
            if metric not in perf_df.columns: continue
            ax = axes[idx]
            perf_df[metric].plot(kind='barh', ax=ax, color=colors)
            ax.set_title(metric, fontweight='bold')
            ax.set_xlim(0.0, 1.0)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Chart saved: {self.output_dir}/model_comparison.png")

    def plot_trade_performance_distribution(self, portfolio_df: pd.DataFrame):
        """Create distribution histogram of trade performances."""
        executed_trades = portfolio_df[portfolio_df['Position'] != 0].copy()

        if len(executed_trades) == 0:
            print("⚠️ No executed trades found for performance distribution.")
            return

        if 'Actual_Return_SL' in executed_trades.columns:
            executed_trades['Actual_Return_Pct'] = executed_trades['Actual_Return_SL']
        else:
            executed_trades['Actual_Return_Pct'] = executed_trades['Actual_Return']

        bins = [-50, -20, -15, -10, -5, -2, 0, 2, 5, 10, 15, 20, 50]
        labels = ['<-20%', '-20% to -15%', '-15% to -10%', '-10% to -5%', '-5% to -2%',
                 '-2% to 0%', '0% to 2%', '2% to 5%', '5% to 10%', '10% to 15%',
                 '15% to 20%', '>20%']

        executed_trades['Return_Bin'] = pd.cut(executed_trades['Actual_Return_Pct'],
                                              bins=bins, labels=labels, include_lowest=True)
        bin_counts = executed_trades['Return_Bin'].value_counts().sort_index()

        plt.figure(figsize=(14, 8))
        colors = ['#e74c3c' if '<' in str(label) or '-' in str(label).split(' to ')[0] else '#27ae60'
                  for label in bin_counts.index]

        bars = plt.bar(range(len(bin_counts)), bin_counts.values,
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1)

        for i, (count, label) in enumerate(zip(bin_counts.values, bin_counts.index)):
            plt.text(i, count + max(bin_counts.values) * 0.01, f'{count}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)

        plt.title('Trade Performance Distribution', fontweight='bold', fontsize=16)
        plt.xlabel('Return Range', fontweight='bold', fontsize=12)
        plt.ylabel('Number of Trades', fontweight='bold', fontsize=12)
        plt.xticks(range(len(bin_counts)), bin_counts.index, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

        total_trades = len(executed_trades)
        winning_trades = len(executed_trades[executed_trades['Actual_Return_Pct'] > 0])
        losing_trades = len(executed_trades[executed_trades['Actual_Return_Pct'] < 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        avg_win = executed_trades[executed_trades['Actual_Return_Pct'] > 0]['Actual_Return_Pct'].mean()
        avg_loss = executed_trades[executed_trades['Actual_Return_Pct'] < 0]['Actual_Return_Pct'].mean()

        avg_win_str = f"{avg_win:.2f}%" if not pd.isna(avg_win) else "N/A"
        avg_loss_str = f"{avg_loss:.2f}%" if not pd.isna(avg_loss) else "N/A"

        stats_text = f"Total Trades: {total_trades}\nWin Rate: {win_rate:.1f}%\nAvg Win: {avg_win_str}\nAvg Loss: {avg_loss_str}"
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/trade_performance_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Trade performance distribution saved: {self.output_dir}/trade_performance_distribution.png")

        print("\n" + "="*80)
        print("📊 TRADE PERFORMANCE BREAKDOWN")
        print("="*80)
        print(f"Total Executed Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades} ({win_rate:.1f}%)")
        print(f"Losing Trades: {losing_trades} ({100-win_rate:.1f}%)")
        print(f"Average Winning Trade: {avg_win_str}")
        print(f"Average Losing Trade: {avg_loss_str}")
        print("\nDetailed Distribution:")
        for label, count in bin_counts.items():
            pct = count / total_trades * 100
            print(f"  {label:<15} | {count:>3} trades ({pct:>5.1f}%)")
        print("="*80 + "\n")
