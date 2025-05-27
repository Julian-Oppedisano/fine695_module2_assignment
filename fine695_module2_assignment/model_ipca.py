# IPCA Model Implementation
# This file will contain the code for the Instrumented Principal Component Analysis model.

import pandas as pd
import numpy as np
import os
import joblib # Potentially for saving IPCA components or a simple final model
import json # Potentially for saving parameters
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression # Example for a simple predictive model on top of IPCA factors
from sklearn.metrics import r2_score
# Need a PCA implementation, sklearn.decomposition.PCA is standard
from sklearn.decomposition import PCA 

from utils import next_window
from portfolio_utils import form_portfolio, calculate_performance_metrics

# Placeholder for the core IPCA logic. This will be complex.
def estimate_ipca_factors_and_predict(train_characteristics, train_returns, test_characteristics, n_factors):
    """
    Estimates Instrumented Principal Components and uses them to predict returns.
    This is a simplified conceptual placeholder.
    A full IPCA implementation (e.g., Kelly, Pruitt, Su) is more involved.

    Args:
        train_characteristics (pd.DataFrame): Training data characteristics (scaled).
        train_returns (pd.Series): Training data returns.
        test_characteristics (pd.DataFrame): Test data characteristics (scaled).
        n_factors (int): Number of principal components / factors to estimate.

    Returns:
        pd.Series: Predictions for the test set.
    """
    # Simplified approach: 
    # 1. Use PCA on characteristics to get principal components (these are not truly "instrumented" yet in a full IPCA sense)
    #    A more IPCA-like approach would involve interacting characteristics with returns or using characteristics to model betas.
    #    For now, let's use PCA on characteristics as a stand-in for factor estimation.
    
    if train_characteristics.shape[0] == 0 or test_characteristics.shape[0] == 0:
        return pd.Series(np.nan, index=test_characteristics.index)
        
    pca = PCA(n_components=n_factors, random_state=42)
    train_pca_factors = pca.fit_transform(train_characteristics)
    test_pca_factors = pca.transform(test_characteristics)

    # 2. Use these factors in a simple linear regression model to predict returns.
    # This model would be trained on train_pca_factors and train_returns.
    if train_pca_factors.shape[0] < 2 : # Not enough samples to train a model
        return pd.Series(np.nan, index=test_characteristics.index)

    factor_model = LinearRegression()
    factor_model.fit(train_pca_factors, train_returns)
    
    predictions = factor_model.predict(test_pca_factors)
    return pd.Series(predictions, index=test_characteristics.index)

def run_ipca_model():
    df = pd.read_csv('Data/homework_sample_big.csv')
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
    df = df.sort_values(['date', 'permno'])
    predictors = pd.read_csv('Data/factors_char_list.csv')['variable'].tolist()
    target = 'stock_exret'

    os.makedirs('outputs', exist_ok=True)

    all_oos_predictions = []
    all_oos_r2 = []
    n_ipca_factors = 5 # Example: Number of IPCA factors to estimate, this could be tuned

    print("Starting IPCA model processing with expanding window...")

    for i, (train_df, val_df, test_df) in enumerate(next_window(df)):
        current_oos_year = test_df['date'].dt.year.iloc[0]
        print(f"Processing OOS Year: {current_oos_year} for IPCA")

        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()

        # Preprocessing: Imputation and Scaling (IPCA often uses scaled characteristics)
        medians = train_df[predictors].median()
        train_df[predictors] = train_df[predictors].fillna(medians)
        val_df[predictors] = val_df[predictors].fillna(medians)
        test_df[predictors] = test_df[predictors].fillna(medians)

        scaler = StandardScaler()
        train_df[predictors] = scaler.fit_transform(train_df[predictors])
        val_df[predictors] = scaler.transform(val_df[predictors])
        test_df[predictors] = scaler.transform(test_df[predictors])
        
        train_df[target] = train_df[target].fillna(0)
        val_df[target] = val_df[target].fillna(0) # Not used directly in this simplified IPCA estimation step for validation
        test_df[target] = test_df[target].fillna(0)

        # For IPCA, "training" is more about estimation on train_df + val_df combined (or just train_df if no hyperparams for IPCA itself)
        # The validation set (val_df) could be used to tune n_ipca_factors if we had a more complex setup.
        # For this version, we'll estimate on train_df and predict on test_df.
        # A more robust approach would estimate on train_df, pick n_factors using val_df, then re-estimate on train+val_df.

        # Combine train and val for IPCA estimation, as per typical expanding window refit
        combined_train_val_df_chars = pd.concat([train_df[predictors], val_df[predictors]])
        combined_train_val_df_returns = pd.concat([train_df[target], val_df[target]])
        
        # Handle cases with insufficient data for PCA or regression
        if combined_train_val_df_chars.shape[0] < n_ipca_factors or combined_train_val_df_chars.empty:
            print(f"Skipping year {current_oos_year} due to insufficient data for IPCA factor estimation.")
            test_df.loc[:, 'ipca_pred'] = np.nan # or some default like 0
        else:
            test_df.loc[:, 'ipca_pred'] = estimate_ipca_factors_and_predict(
                combined_train_val_df_chars,
                combined_train_val_df_returns,
                test_df[predictors],
                n_factors=n_ipca_factors
            )
        
        # Fill any NaNs in predictions with 0 (e.g., if a stock had all NaN predictors)
        test_df['ipca_pred'] = test_df['ipca_pred'].fillna(0)

        oos_r2_year = r2_score(test_df[target], test_df['ipca_pred'])
        all_oos_r2.append(oos_r2_year)
        print(f"Year {current_oos_year} - IPCA OOS R2: {oos_r2_year:.4f}")

        all_oos_predictions.append(test_df[['date', 'permno', 'ipca_pred', target]])
        
        # Note: Saving IPCA model components (like PCA transformations or factor loadings)
        # would be more complex than a simple sklearn model. For now, not saving them.

    print("Finished processing all windows for IPCA model.")

    if not all_oos_predictions:
        print("No IPCA predictions were generated. Exiting.")
        return

    all_predictions_df = pd.concat(all_oos_predictions)
    overall_oos_r2 = np.mean(all_oos_r2) if all_oos_r2 else np.nan
    print(f'Overall Average OOS R² score for IPCA: {overall_oos_r2:.4f}')

    metrics_df = pd.DataFrame({'model': ['ipca'], 'oos_r2': [overall_oos_r2]})
    metrics_file = 'outputs/metrics.csv'
    if os.path.exists(metrics_file):
        existing_metrics = pd.read_csv(metrics_file)
        existing_metrics = existing_metrics[existing_metrics['model'] != 'ipca']
        metrics_df = pd.concat([existing_metrics, metrics_df])
    metrics_df.to_csv(metrics_file, index=False)
    print(f'Overall IPCA metrics saved to {metrics_file}')

    all_predictions_df = all_predictions_df.rename(columns={target: 'stock_exret'})
    ipca_overall_port_ret = form_portfolio(all_predictions_df, 'ipca_pred', n_stocks=50)
    ipca_overall_port_ret.columns = ['ipca_port_ret']
    ipca_overall_port_ret.to_csv('outputs/ipca_overall_port_ret.csv')
    print('Overall IPCA monthly portfolio returns saved to outputs/ipca_overall_port_ret.csv')

    spy_returns = pd.read_excel('Data/SPY returns.xlsx', index_col=0)
    spy_returns.index = pd.to_datetime(spy_returns.index)
    perf_summary_df = calculate_performance_metrics(ipca_overall_port_ret, spy_returns, 'ipca')
    summary_file = 'outputs/perf_summary.csv'
    if os.path.exists(summary_file):
        existing_summary = pd.read_csv(summary_file)
        existing_summary = existing_summary[existing_summary['model'] != 'ipca']
        perf_summary_df = pd.concat([existing_summary, perf_summary_df])
    perf_summary_df.to_csv(summary_file, index=False)
    print(f'Overall IPCA performance summary saved to {summary_file}')

if __name__ == "__main__":
    run_ipca_model() 