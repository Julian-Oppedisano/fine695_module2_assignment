import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import joblib
import json # For parameters, though RidgeCV stores alpha_
import os
from utils import next_window
from portfolio_utils import form_portfolio, calculate_performance_metrics

def run_ridge_model():
    # Load data and predictors
    df = pd.read_csv('Data/homework_sample_big.csv')
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
    df = df.sort_values(['date', 'permno'])
    predictors = pd.read_csv('Data/factors_char_list.csv')['variable'].tolist()
    target = 'stock_exret'

    os.makedirs('outputs', exist_ok=True)

    all_oos_predictions = []
    all_oos_r2 = []
    
    ridge_alphas = [0.1, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0] # Expanded alphas slightly

    print("Starting Ridge model processing with expanding window (with scaling)...")

    for i, (train_df, val_df, test_df) in enumerate(next_window(df)):
        current_oos_year = test_df['date'].dt.year.iloc[0]
        print(f"Processing OOS Year: {current_oos_year} for Ridge")

        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()

        medians = train_df[predictors].median()
        train_df[predictors] = train_df[predictors].fillna(medians)
        val_df[predictors] = val_df[predictors].fillna(medians)
        test_df[predictors] = test_df[predictors].fillna(medians)
        
        train_df[target] = train_df[target].fillna(0)
        val_df[target] = val_df[target].fillna(0)
        test_df[target] = test_df[target].fillna(0)

        # Create a pipeline with StandardScaler and RidgeCV
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', RidgeCV(alphas=ridge_alphas, store_cv_results=True)) # store_cv_results is the new name
        ])
        
        # Fit the pipeline on train_df. RidgeCV will select best alpha based on internal CV on scaled data.
        pipeline.fit(train_df[predictors], train_df[target])
        best_alpha = pipeline.named_steps['ridge'].alpha_

        # Re-fit on combined train_df + val_df using the best alpha and same scaling
        # The pipeline handles fitting the scaler on the new combined data and then fitting Ridge.
        final_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', RidgeCV(alphas=[best_alpha], store_cv_results=True)) # Effectively a Ridge model with fixed alpha
        ])

        combined_train_val_df = pd.concat([train_df, val_df])
        final_pipeline.fit(combined_train_val_df[predictors], combined_train_val_df[target])

        test_df.loc[:, 'ridge_pred'] = final_pipeline.predict(test_df[predictors])
        
        oos_r2_year = r2_score(test_df[target], test_df['ridge_pred'])
        all_oos_r2.append(oos_r2_year)
        print(f"Year {current_oos_year} - Ridge OOS R2: {oos_r2_year:.4f}, Best Alpha: {best_alpha}")

        all_oos_predictions.append(test_df[['date', 'permno', 'ridge_pred', target]])

        if i == len(list(next_window(df))) - 1: 
            joblib.dump(final_pipeline, 'outputs/ridge_model_last_window.pkl')
            print(f"Saved final Ridge pipeline (alpha={best_alpha}) for last window to outputs/ridge_model_last_window.pkl")
            params_to_save = {'best_alpha_last_window': best_alpha, 'oos_r2_last_window': oos_r2_year}
            with open('outputs/ridge_params_last_window.json', 'w') as f:
                json.dump(params_to_save, f)
            print("Saved params for Ridge (last window) to outputs/ridge_params_last_window.json")
            
    print("Finished processing all windows for Ridge model.")

    if not all_oos_predictions:
        print("No Ridge predictions were generated. Exiting.")
        return

    all_predictions_df = pd.concat(all_oos_predictions)
    overall_oos_r2 = np.mean(all_oos_r2) if all_oos_r2 else np.nan
    print(f'Overall Average OOS R² score for Ridge: {overall_oos_r2:.4f}')

    metrics_df = pd.DataFrame({'model': ['ridge'], 'oos_r2': [overall_oos_r2]})
    metrics_file = 'outputs/metrics.csv'
    if os.path.exists(metrics_file):
        existing_metrics = pd.read_csv(metrics_file)
        existing_metrics = existing_metrics[existing_metrics['model'] != 'ridge']
        metrics_df = pd.concat([existing_metrics, metrics_df])
    metrics_df.to_csv(metrics_file, index=False)
    print(f'Overall Ridge metrics saved to {metrics_file}')

    all_predictions_df = all_predictions_df.rename(columns={target: 'stock_exret'})
    ridge_overall_port_ret = form_portfolio(all_predictions_df, 'ridge_pred', n_stocks=50)
    ridge_overall_port_ret.columns = ['ridge_port_ret'] 
    ridge_overall_port_ret.to_csv('outputs/ridge_overall_port_ret.csv')
    print('Overall Ridge monthly portfolio returns saved to outputs/ridge_overall_port_ret.csv')

    spy_returns = pd.read_excel('Data/SPY returns.xlsx', index_col=0)
    spy_returns.index = pd.to_datetime(spy_returns.index)
    perf_summary_df = calculate_performance_metrics(ridge_overall_port_ret, spy_returns, 'ridge')
    summary_file = 'outputs/perf_summary.csv'
    if os.path.exists(summary_file):
        existing_summary = pd.read_csv(summary_file)
        existing_summary = existing_summary[existing_summary['model'] != 'ridge']
        perf_summary_df = pd.concat([existing_summary, perf_summary_df])
    perf_summary_df.to_csv(summary_file, index=False)
    print(f'Overall Ridge performance summary saved to {summary_file}')

if __name__ == "__main__":
    run_ridge_model() 