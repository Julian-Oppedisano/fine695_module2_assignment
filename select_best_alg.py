import pandas as pd
import numpy as np

# Read performance summary
perf_summary = pd.read_csv('outputs/perf_summary.csv')

# Remove duplicate entries by keeping the last entry for each model
perf_summary = perf_summary.dropna(subset=['alpha', 'sharpe']).groupby('model').last().reset_index()

# Filter out unrealistic values (e.g., alpha > 1 or Sharpe > 10)
perf_summary = perf_summary[
    (perf_summary['alpha'].abs() <= 1) & 
    (perf_summary['sharpe'].abs() <= 10)
]

# Read SPY returns to calculate its Sharpe ratio
spy_returns = pd.read_excel('Data/SPY returns.xlsx', index_col=0)
spy_returns.index = pd.to_datetime(spy_returns.index)
spy_sharpe = (spy_returns['SPY_ret'].mean() / spy_returns['SPY_ret'].std()) * (12 ** 0.5)

print("\nPerformance Summary:")
print(perf_summary)
print(f"\nSPY Sharpe Ratio: {spy_sharpe:.4f}")

# Filter algorithms with positive alpha and Sharpe > SPY
valid_models = perf_summary[
    (perf_summary['alpha'] > 0) & 
    (perf_summary['sharpe'] > spy_sharpe)
]

if len(valid_models) > 0:
    # Select model with highest Sharpe ratio
    best_model = valid_models.loc[valid_models['sharpe'].idxmax()]
    best_alg = best_model['model']
    print(f"\nBest algorithm: {best_alg}")
    print(f"Sharpe ratio: {best_model['sharpe']:.4f}")
    print(f"Alpha: {best_model['alpha']:.4f}")
else:
    # If no model meets criteria, select the one with highest Sharpe
    best_model = perf_summary.loc[perf_summary['sharpe'].idxmax()]
    best_alg = best_model['model']
    print("\nNo model meets both criteria (positive alpha and Sharpe > SPY)")
    print(f"Selecting model with highest Sharpe: {best_alg}")
    print(f"Sharpe ratio: {best_model['sharpe']:.4f}")
    print(f"Alpha: {best_model['alpha']:.4f}")

# Save best algorithm name to file
with open('outputs/best_alg.txt', 'w') as f:
    f.write(best_alg)
print(f"\nBest algorithm saved to outputs/best_alg.txt") 