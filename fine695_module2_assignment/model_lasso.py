import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import joblib
import json
import os
import numpy as np # Added for np.mean
from utils import next_window
from portfolio_utils import form_portfolio, calculate_performance_metrics

def run_lasso_model():
    # Load data and predictors
    df = pd.read_csv('Data/homework_sample_big.csv')
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
    df = df.sort_values(['date', 'permno'])
    predictors = pd.read_csv('Data/factors_char_list.csv')['variable'].tolist()
    target = 'stock_exret'

    os.makedirs('outputs', exist_ok=True)

    all_oos_predictions = []
    all_oos_r2 = []
    
    print("Starting Lasso model processing with expanding window...")

    for i, (train_df, val_df, test_df) in enumerate(next_window(df)):
        current_oos_year = test_df['date'].dt.year.iloc[0]
        print(f"Processing OOS Year: {current_oos_year}")

        # Ensure dataframes are copies to avoid SettingWithCopyWarning
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()

        # Median imputation for the current window
        # Important: Calculate medians only on the current training set
        medians = train_df[predictors].median()
        train_df[predictors] = train_df[predictors].fillna(medians)
        val_df[predictors] = val_df[predictors].fillna(medians)
        test_df[predictors] = test_df[predictors].fillna(medians)
        
        # Fill any remaining NaNs in target (e.g. if a stock has no return data)
        train_df[target] = train_df[target].fillna(0)
        val_df[target] = val_df[target].fillna(0)
        test_df[target] = test_df[target].fillna(0)


        # Define pipeline and grid for GridSearchCV
        pipeline = Pipeline([
            ('model', Lasso(max_iter=10000, tol=0.001)) # Increased tol for faster convergence
        ])
        grid = {
            'model__alpha': [0.001, 0.01, 0.1] # Adjusted alpha range
        }

        # GridSearchCV on current train_df, score=r2
        # Using a simpler CV split for speed in each window; could be TimeSeriesSplit
        search = GridSearchCV(pipeline, grid, scoring='r2', cv=3, n_jobs=-1)
        search.fit(train_df[predictors], train_df[target])

        # Optional: Evaluate on validation split for this window (for logging or early stopping if implemented)
        # y_val_pred = search.predict(val_df[predictors])
        # val_r2 = r2_score(val_df[target], y_val_pred)
        # print(f"Year {current_oos_year} - Validation R2: {val_r2:.4f}, Best Alpha: {search.best_params_['model__alpha']}")

        # Re-fit on combined train_df + val_df using best params from this window's search
        best_alpha = search.best_params_['model__alpha']
        final_model = Lasso(alpha=best_alpha, max_iter=10000, tol=0.001)
        
        combined_train_val_df = pd.concat([train_df, val_df])
        final_model.fit(combined_train_val_df[predictors], combined_train_val_df[target])

        # Generate predictions on the current test_df
        test_df.loc[:, 'lasso_pred'] = final_model.predict(test_df[predictors])
        
        # Calculate and store OOS R² for the current year
        oos_r2_year = r2_score(test_df[target], test_df['lasso_pred'])
        all_oos_r2.append(oos_r2_year)
        print(f"Year {current_oos_year} - OOS R2: {oos_r2_year:.4f}")

        # Store predictions for this year (date, permno, prediction, actual_return)
        all_oos_predictions.append(test_df[['date', 'permno', 'lasso_pred', target]])

        # Optional: Save model and params for the last window or each window
        if i == len(list(next_window(df))) - 1: # If it's the last window
            joblib.dump(final_model, 'outputs/lasso_model_last_window.pkl')
            print(f"Saved final Lasso model (alpha={best_alpha}) for last window to outputs/lasso_model_last_window.pkl")
            params_to_save = {'best_alpha_last_window': best_alpha, 'oos_r2_last_window': oos_r2_year}
            with open('outputs/lasso_params_last_window.json', 'w') as f:
                json.dump(params_to_save, f)
            print("Saved params for last window to outputs/lasso_params_last_window.json")
            
    print("Finished processing all windows for Lasso model.")

    if not all_oos_predictions:
        print("No predictions were generated. Exiting.")
        return

    # Combine all OOS predictions
    all_predictions_df = pd.concat(all_oos_predictions)

    # Calculate overall OOS R² (average of yearly R²)
    overall_oos_r2 = np.mean(all_oos_r2) if all_oos_r2 else np.nan
    print(f'Overall Average OOS R² score for Lasso: {overall_oos_r2:.4f}')

    # Save overall OOS R² to metrics.csv
    metrics_df = pd.DataFrame({'model': ['lasso'], 'oos_r2': [overall_oos_r2]})
    # Overwrite if exists, or append if running multiple models serially and want one file
    metrics_file = 'outputs/metrics.csv'
    if os.path.exists(metrics_file):
        existing_metrics = pd.read_csv(metrics_file)
        existing_metrics = existing_metrics[existing_metrics['model'] != 'lasso'] # Remove old lasso entry
        metrics_df = pd.concat([existing_metrics, metrics_df])
    metrics_df.to_csv(metrics_file, index=False)
    print(f'Overall Lasso metrics saved to {metrics_file}')

    # Form monthly portfolio returns across all OOS years
    # Ensure 'stock_exret' is present for form_portfolio
    all_predictions_df = all_predictions_df.rename(columns={target: 'stock_exret'})
    lasso_overall_port_ret = form_portfolio(all_predictions_df, 'lasso_pred', n_stocks=50)
    
    # Save combined portfolio returns
    lasso_overall_port_ret.columns = ['lasso_port_ret'] 
    lasso_overall_port_ret.to_csv('outputs/lasso_overall_port_ret.csv')
    print('Overall Lasso monthly portfolio returns saved to outputs/lasso_overall_port_ret.csv')

    # Calculate and save overall performance summary
    spy_returns = pd.read_excel('Data/SPY returns.xlsx', index_col=0)
    spy_returns.index = pd.to_datetime(spy_returns.index)
    
    perf_summary_df = calculate_performance_metrics(lasso_overall_port_ret, spy_returns, 'lasso')
    
    summary_file = 'outputs/perf_summary.csv'
    if os.path.exists(summary_file):
        existing_summary = pd.read_csv(summary_file)
        existing_summary = existing_summary[existing_summary['model'] != 'lasso'] # Remove old lasso entry
        perf_summary_df = pd.concat([existing_summary, perf_summary_df])
    perf_summary_df.to_csv(summary_file, index=False)
    print(f'Overall Lasso performance summary saved to {summary_file}')

if __name__ == "__main__":
    run_lasso_model() 