import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import joblib
import json
import os
from utils import next_window
from portfolio_utils import form_portfolio, calculate_performance_metrics

def run_elasticnet_model():
    df = pd.read_csv('Data/homework_sample_big.csv')
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
    df = df.sort_values(['date', 'permno'])
    predictors = pd.read_csv('Data/factors_char_list.csv')['variable'].tolist()
    target = 'stock_exret'

    os.makedirs('outputs', exist_ok=True)

    all_oos_predictions = []
    all_oos_r2 = []
    
    enet_l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    enet_alphas = [0.0001, 0.001, 0.01, 0.1, 1.0]

    print("Starting ElasticNet model processing with expanding window (with scaling)...")

    for i, (train_df, val_df, test_df) in enumerate(next_window(df)):
        current_oos_year = test_df['date'].dt.year.iloc[0]
        print(f"Processing OOS Year: {current_oos_year} for ElasticNet")

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

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('elasticnet', ElasticNetCV(
                l1_ratio=enet_l1_ratios, 
                alphas=enet_alphas, 
                cv=3, 
                n_jobs=-1, 
                max_iter=2000, 
                tol=0.001,
                random_state=42
            ))
        ])
        
        pipeline.fit(train_df[predictors], train_df[target])
        best_alpha = pipeline.named_steps['elasticnet'].alpha_
        best_l1_ratio = pipeline.named_steps['elasticnet'].l1_ratio_

        final_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('elasticnet', ElasticNetCV(
                l1_ratio=[best_l1_ratio], 
                alphas=[best_alpha], 
                cv=3,
                n_jobs=-1, 
                max_iter=2000,
                tol=0.001,
                random_state=42
            ))
        ])

        combined_train_val_df = pd.concat([train_df, val_df])
        final_pipeline.fit(combined_train_val_df[predictors], combined_train_val_df[target])

        test_df.loc[:, 'elasticnet_pred'] = final_pipeline.predict(test_df[predictors])
        
        oos_r2_year = r2_score(test_df[target], test_df['elasticnet_pred'])
        all_oos_r2.append(oos_r2_year)
        print(f"Year {current_oos_year} - ElasticNet OOS R2: {oos_r2_year:.4f}, Alpha: {best_alpha:.4f}, L1 Ratio: {best_l1_ratio:.2f}")

        all_oos_predictions.append(test_df[['date', 'permno', 'elasticnet_pred', target]])

        if i == len(list(next_window(df))) - 1: 
            joblib.dump(final_pipeline, 'outputs/elasticnet_model_last_window.pkl')
            print(f"Saved final ElasticNet pipeline for last window to outputs/elasticnet_model_last_window.pkl")
            params_to_save = {
                'best_alpha_last_window': best_alpha,
                'best_l1_ratio_last_window': best_l1_ratio,
                'oos_r2_last_window': oos_r2_year
            }
            with open('outputs/elasticnet_params_last_window.json', 'w') as f:
                json.dump(params_to_save, f)
            print("Saved params for ElasticNet (last window) to outputs/elasticnet_params_last_window.json")
            
    print("Finished processing all windows for ElasticNet model.")

    if not all_oos_predictions:
        print("No ElasticNet predictions were generated. Exiting.")
        return

    all_predictions_df = pd.concat(all_oos_predictions)
    overall_oos_r2 = np.mean(all_oos_r2) if all_oos_r2 else np.nan
    print(f'Overall Average OOS R² score for ElasticNet: {overall_oos_r2:.4f}')

    metrics_df = pd.DataFrame({'model': ['elasticnet'], 'oos_r2': [overall_oos_r2]})
    metrics_file = 'outputs/metrics.csv'
    if os.path.exists(metrics_file):
        existing_metrics = pd.read_csv(metrics_file)
        existing_metrics = existing_metrics[existing_metrics['model'] != 'elasticnet']
        metrics_df = pd.concat([existing_metrics, metrics_df])
    metrics_df.to_csv(metrics_file, index=False)
    print(f'Overall ElasticNet metrics saved to {metrics_file}')

    all_predictions_df = all_predictions_df.rename(columns={target: 'stock_exret'})
    elasticnet_overall_port_ret = form_portfolio(all_predictions_df, 'elasticnet_pred', n_stocks=50)
    elasticnet_overall_port_ret.columns = ['elasticnet_port_ret'] 
    elasticnet_overall_port_ret.to_csv('outputs/elasticnet_overall_port_ret.csv')
    print('Overall ElasticNet monthly portfolio returns saved to outputs/elasticnet_overall_port_ret.csv')

    spy_returns = pd.read_excel('Data/SPY returns.xlsx', index_col=0)
    spy_returns.index = pd.to_datetime(spy_returns.index)
    perf_summary_df = calculate_performance_metrics(elasticnet_overall_port_ret, spy_returns, 'elasticnet')
    summary_file = 'outputs/perf_summary.csv'
    if os.path.exists(summary_file):
        existing_summary = pd.read_csv(summary_file)
        existing_summary = existing_summary[existing_summary['model'] != 'elasticnet']
        perf_summary_df = pd.concat([existing_summary, perf_summary_df])
    perf_summary_df.to_csv(summary_file, index=False)
    print(f'Overall ElasticNet performance summary saved to {summary_file}')

if __name__ == "__main__":
    run_elasticnet_model() 