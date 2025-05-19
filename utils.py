import pandas as pd
import numpy as np

def next_window(df):
    """
    Generator yielding (train, val, test) DataFrames for each yearly expanding window.
    Assumes df has a 'date' column of type datetime.
    """
    for year in range(2017, 2024):  # Adjust end year as needed
        train_end = pd.to_datetime(f"{year-3}-12-31")
        val_end = pd.to_datetime(f"{year-1}-12-31")
        test_start = pd.to_datetime(f"{year}-01-01")
        test_end = pd.to_datetime(f"{year}-12-31")

        train = df[df['date'] <= train_end]
        val = df[(df['date'] > train_end) & (df['date'] <= val_end)]
        test = df[(df['date'] >= test_start) & (df['date'] <= test_end)]

        if len(train) > 0 and len(val) > 0 and len(test) > 0:
            yield train, val, test 

def validate_time_windows(df):
    """
    Validate that the data splits follow the required time windows:
    - First 10 years for training
    - Next 2 years for validation
    - Subsequent year for OOS test
    
    Args:
        df (pd.DataFrame): DataFrame with 'date' column
        
    Returns:
        bool: True if time windows are correct
    """
    dates = pd.to_datetime(df['date'].unique())
    dates = sorted(dates)
    
    # Calculate expected splits
    train_end = dates[0] + pd.DateOffset(years=10)
    val_end = train_end + pd.DateOffset(years=2)
    test_end = val_end + pd.DateOffset(years=1)
    
    # Get actual splits
    train_dates = dates[dates <= train_end]
    val_dates = dates[(dates > train_end) & (dates <= val_end)]
    test_dates = dates[(dates > val_end) & (dates <= test_end)]
    
    # Validate
    if len(train_dates) < 120:  # 10 years * 12 months
        raise ValueError("Training period must be at least 10 years")
    if len(val_dates) < 24:  # 2 years * 12 months
        raise ValueError("Validation period must be at least 2 years")
    if len(test_dates) < 12:  # 1 year * 12 months
        raise ValueError("Test period must be at least 1 year")
        
    return True

def validate_portfolio(portfolio):
    """
    Validate that the portfolio meets requirements:
    - Minimum 50 stocks
    - Equal weights sum to 1
    
    Args:
        portfolio (pd.DataFrame): Portfolio weights DataFrame
        
    Returns:
        bool: True if portfolio meets requirements
    """
    if len(portfolio) < 50:
        raise ValueError("Portfolio must contain at least 50 stocks")
    
    weights_sum = portfolio['weight'].sum()
    if not np.isclose(weights_sum, 1.0, atol=1e-6):
        raise ValueError(f"Portfolio weights must sum to 1, got {weights_sum}")
        
    return True 