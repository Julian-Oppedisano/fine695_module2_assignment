import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_spy_metrics(spy_returns):
    """
    Calculate Sharpe ratio for SPY returns.
    
    Args:
        spy_returns (pd.DataFrame): SPY returns with datetime index
        
    Returns:
        float: Annualized Sharpe ratio for SPY
    """
    try:
        # Ensure index is datetime
        spy_returns.index = pd.to_datetime(spy_returns.index)
        
        # Calculate annualized Sharpe ratio
        spy_sharpe = (spy_returns['SPY_ret'].mean() / spy_returns['SPY_ret'].std()) * (12 ** 0.5)
        
        return spy_sharpe
        
    except Exception as e:
        logger.error(f"Error calculating SPY metrics: {str(e)}")
        raise

def clean_performance_summary(perf_summary):
    """
    Clean the performance summary DataFrame.
    
    Args:
        perf_summary (pd.DataFrame): Raw performance summary
        
    Returns:
        pd.DataFrame: Cleaned performance summary
    """
    try:
        # Remove duplicate rows
        perf_summary = perf_summary.drop_duplicates()
        
        # Remove rows with any NaN values
        perf_summary = perf_summary.dropna()
        
        # Remove rows with unrealistic values
        # Check for extremely high values (e.g., > 1000)
        for col in ['alpha', 'sharpe', 'stdev']:
            perf_summary = perf_summary[perf_summary[col] < 1000]
            
        # Check for negative max_drawdown
        perf_summary = perf_summary[perf_summary['max_drawdown'] >= 0]
        
        return perf_summary
        
    except Exception as e:
        logger.error(f"Error cleaning performance summary: {str(e)}")
        raise

def select_best_model(perf_summary_path, spy_returns_path):
    """
    Select the best model based on performance metrics.
    
    Args:
        perf_summary_path (str): Path to performance summary CSV
        spy_returns_path (str): Path to SPY returns Excel file
        
    Returns:
        str: Name of the best model
    """
    try:
        # Read performance summary
        perf_summary = pd.read_csv(perf_summary_path)
        
        # Clean performance summary
        perf_summary = clean_performance_summary(perf_summary)
        
        if len(perf_summary) == 0:
            raise ValueError("No valid models found after cleaning performance summary")
            
        # Read SPY returns
        spy_returns = pd.read_excel(spy_returns_path, index_col=0)
        spy_returns.index = pd.to_datetime(spy_returns.index)
        
        # Calculate SPY Sharpe ratio
        spy_sharpe = calculate_spy_metrics(spy_returns)
        logger.info(f"SPY Sharpe ratio: {spy_sharpe:.4f}")
        
        # Filter models with positive alpha and Sharpe > SPY
        valid_models = perf_summary[
            (perf_summary['alpha'] > 0) & 
            (perf_summary['sharpe'] > spy_sharpe)
        ]
        
        if len(valid_models) == 0:
            logger.warning("No models meet the criteria (positive alpha and Sharpe > SPY)")
            # Select model with highest Sharpe ratio regardless of criteria
            best_model = perf_summary.loc[perf_summary['sharpe'].idxmax(), 'model']
            logger.info(f"Selected model with highest Sharpe ratio: {best_model}")
        else:
            # Select model with highest Sharpe ratio among valid models
            best_model = valid_models.loc[valid_models['sharpe'].idxmax(), 'model']
            logger.info(f"Selected best model: {best_model}")
            
        # Print detailed metrics for the best model
        best_metrics = perf_summary[perf_summary['model'] == best_model].iloc[0]
        logger.info("\nBest model metrics:")
        logger.info(f"Model: {best_metrics['model']}")
        logger.info(f"Alpha: {best_metrics['alpha']:.4f}")
        logger.info(f"Sharpe: {best_metrics['sharpe']:.4f}")
        logger.info(f"Std Dev: {best_metrics['stdev']:.4f}")
        logger.info(f"Max Drawdown: {best_metrics['max_drawdown']:.4f}")
        logger.info(f"Max Loss: {best_metrics['max_loss']:.4f}")
        
        # Save best model name to file
        with open('outputs/best_alg.txt', 'w') as f:
            f.write(best_model)
        logger.info(f"Best model name saved to outputs/best_alg.txt")
        
        return best_model
        
    except Exception as e:
        logger.error(f"Error selecting best model: {str(e)}")
        raise

if __name__ == "__main__":
    # Select best model
    best_model = select_best_model(
        perf_summary_path='outputs/perf_summary.csv',
        spy_returns_path='Data/SPY returns.xlsx'
    ) 