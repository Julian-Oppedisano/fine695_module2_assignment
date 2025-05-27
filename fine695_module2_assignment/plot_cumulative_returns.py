import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Paths
BEST_ALG_PATH = 'outputs/best_alg.txt'
PORT_RET_DIR = 'outputs'
SPY_RET_PATH = 'Data/SPY returns.xlsx'
CUM_RET_PNG = 'outputs/cum_returns.png'

# Read best model name
def get_best_model_name():
    with open(BEST_ALG_PATH, 'r') as f:
        return f.read().strip()

def load_portfolio_returns(model_name):
    port_ret_path = os.path.join(PORT_RET_DIR, f'{model_name}_overall_port_ret.csv')
    port_ret = pd.read_csv(port_ret_path, index_col=0, parse_dates=True)
    # Use the correct column name
    col = [c for c in port_ret.columns if c.startswith(model_name)][0] if any(c.startswith(model_name) for c in port_ret.columns) else port_ret.columns[0]
    port_ret = port_ret.rename(columns={col: 'port_ret'})
    return port_ret

def load_spy_returns():
    spy = pd.read_excel(SPY_RET_PATH, index_col=0)
    spy.index = pd.to_datetime(spy.index)
    if 'SPY_ret' not in spy.columns:
        # Try to infer the column
        spy.columns = ['SPY_ret']
    return spy[['SPY_ret']]

def align_and_cumulate(port_ret, spy_ret):
    # Align to month end
    port_ret.index = port_ret.index.to_period('M').to_timestamp('M')
    spy_ret.index = spy_ret.index.to_period('M').to_timestamp('M')
    # Merge
    merged = pd.concat([port_ret, spy_ret], axis=1, join='inner')
    merged = merged.dropna()
    # Compute cumulative returns
    merged['cum_port'] = (1 + merged['port_ret']).cumprod() - 1
    merged['cum_spy'] = (1 + merged['SPY_ret']).cumprod() - 1
    return merged

def plot_cumulative_returns(merged, model_name):
    plt.figure(figsize=(10,6))
    plt.plot(merged.index, merged['cum_port'], label=f'{model_name} Portfolio', linewidth=2)
    plt.plot(merged.index, merged['cum_spy'], label='SPY', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title(f'Cumulative OOS Returns: {model_name} vs SPY')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(CUM_RET_PNG, dpi=150)
    plt.close()
    print(f'Cumulative returns plot saved to {CUM_RET_PNG}')

if __name__ == '__main__':
    model_name = get_best_model_name()
    port_ret = load_portfolio_returns(model_name)
    spy_ret = load_spy_returns()
    merged = align_and_cumulate(port_ret, spy_ret)
    plot_cumulative_returns(merged, model_name) 