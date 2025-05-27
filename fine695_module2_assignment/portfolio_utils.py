import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def form_portfolio(df, pred_col, n_stocks=50):
    """
    Form an equally-weighted portfolio of top n_stocks based on predictions.
    
    Args:
        df (pd.DataFrame): DataFrame containing predictions and returns
        pred_col (str): Name of the prediction column
        n_stocks (int): Number of stocks to include in portfolio
        
    Returns:
        pd.DataFrame: Monthly portfolio returns with a single column named 'port_ret'
        
    Raises:
        ValueError: If input data is invalid or groupby/apply result is unexpected
    """
    try:
        # Validate inputs
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if pred_col not in df.columns:
            raise ValueError(f"Prediction column '{pred_col}' not found in DataFrame")
        if 'stock_exret' not in df.columns:
            raise ValueError("'stock_exret' column not found in DataFrame")
        if 'date' not in df.columns:
            raise ValueError("'date' column not found in DataFrame")
            
        # Rank predictions within each date
        df['rank'] = df.groupby('date')[pred_col].rank(ascending=False)
        
        # Select top n_stocks
        top_n = df[df['rank'] <= n_stocks].copy()
        
        if len(top_n) == 0:
            raise ValueError("No stocks selected after ranking")
            
        # Calculate equal weights
        weights = 1 / n_stocks
        top_n['weight'] = weights
        
        # Calculate portfolio returns
        port_ret = top_n.groupby('date', group_keys=False).apply(
            lambda x: (x['stock_exret'] * x['weight']).sum()
        )
        
        # Ensure we return a DataFrame with a single column
        if isinstance(port_ret, pd.Series):
            port_ret = port_ret.to_frame()
        
        # Verify the result has exactly one column
        if len(port_ret.columns) != 1:
            raise ValueError(f"Unexpected number of columns in result: {len(port_ret.columns)}")
            
        # Set the column name
        port_ret.columns = ['port_ret']
        
        return port_ret
        
    except Exception as e:
        logger.error(f"Error in form_portfolio: {str(e)}")
        raise

def calculate_performance_metrics(port_ret, spy_returns, model_name):
    """
    Calculate performance metrics for a portfolio.
    
    Args:
        port_ret (pd.DataFrame): Portfolio returns
        spy_returns (pd.DataFrame): SPY returns
        model_name (str): Name of the model
        
    Returns:
        pd.DataFrame: Performance metrics
        
    Raises:
        ValueError: If input data is invalid or calculations fail
    """
    try:
        # Validate inputs
        if not isinstance(port_ret, pd.DataFrame):
            raise ValueError("Portfolio returns must be a pandas DataFrame")
        if not isinstance(spy_returns, pd.DataFrame):
            raise ValueError("SPY returns must be a pandas DataFrame")
            
        # Ensure indices are datetime
        port_ret.index = pd.to_datetime(port_ret.index)
        spy_returns.index = pd.to_datetime(spy_returns.index)
        
        # Align indices to month end
        port_ret.index = port_ret.index.to_period('M').to_timestamp('M')
        spy_returns.index = spy_returns.index.to_period('M').to_timestamp('M')
        
        # Merge returns
        merged_returns = pd.concat([port_ret, spy_returns], axis=1)
        
        if len(merged_returns) == 0:
            raise ValueError("No overlapping dates between portfolio and SPY returns")
            
        merged_returns.columns = [f'{model_name}_port_ret', 'spy_ret']
        
        # Calculate metrics with error handling
        try:
            alpha = merged_returns[f'{model_name}_port_ret'].mean() - merged_returns['spy_ret'].mean()
        except Exception as e:
            logger.warning(f"Error calculating alpha: {str(e)}")
            alpha = np.nan
            
        try:
            std = merged_returns[f'{model_name}_port_ret'].std()
            if std == 0:
                logger.warning("Portfolio standard deviation is zero")
                sharpe = np.nan
            else:
                sharpe = (merged_returns[f'{model_name}_port_ret'].mean() / std) * (12 ** 0.5)
        except Exception as e:
            logger.warning(f"Error calculating Sharpe ratio: {str(e)}")
            sharpe = np.nan
            
        try:
            stdev = merged_returns[f'{model_name}_port_ret'].std() * (12 ** 0.5)
        except Exception as e:
            logger.warning(f"Error calculating standard deviation: {str(e)}")
            stdev = np.nan
            
        try:
            max_drawdown = (merged_returns[f'{model_name}_port_ret'].cummax() - merged_returns[f'{model_name}_port_ret']).max()
        except Exception as e:
            logger.warning(f"Error calculating max drawdown: {str(e)}")
            max_drawdown = np.nan
            
        try:
            max_loss = merged_returns[f'{model_name}_port_ret'].min()
        except Exception as e:
            logger.warning(f"Error calculating max loss: {str(e)}")
            max_loss = np.nan
        
        # Create summary DataFrame
        perf_summary = pd.DataFrame({
            'model': [model_name],
            'alpha': [alpha],
            'sharpe': [sharpe],
            'stdev': [stdev],
            'max_drawdown': [max_drawdown],
            'max_loss': [max_loss]
        })
        
        return perf_summary
        
    except Exception as e:
        logger.error(f"Error in calculate_performance_metrics: {str(e)}")
        raise 