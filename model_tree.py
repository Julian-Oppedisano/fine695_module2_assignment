import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import joblib
import json
import os
from utils import next_window
from portfolio_utils import form_portfolio, calculate_performance_metrics

def run_tree_model():
    df = pd.read_csv('Data/homework_sample_big.csv')
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
    df = df.sort_values(['date', 'permno'])
    predictors = pd.read_csv('Data/factors_char_list.csv')['variable'].tolist()
    target = 'stock_exret'

    os.makedirs('outputs', exist_ok=True)

    all_oos_predictions = []
    all_oos_r2 = []
    
    # Decision Tree parameters for GridSearchCV
    # Simple grid for speed. Can be expanded.
    tree_param_grid = {
        'model__max_depth': [5, 10, None],
        'model__min_samples_leaf': [10, 20, 50]
    }

    print("Starting Decision Tree model processing with expanding window...")

    for i, (train_df, val_df, test_df) in enumerate(next_window(df)):
        current_oos_year = test_df['date'].dt.year.iloc[0]
        print(f"Processing OOS Year: {current_oos_year} for Decision Tree")

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

        # Pipeline for Decision Tree (optional, but good practice)
        # No scaler needed for basic decision trees, but pipeline helps keep structure consistent.
        pipeline = Pipeline([
            ('model', DecisionTreeRegressor(random_state=42))
        ])

        # GridSearchCV on current train_df, using val_df for scoring in the search if desired,
        # or use internal CV splits from train_df.
        # For simplicity with expanding window, using CV on train_df, then evaluate on val_df separately.
        search = GridSearchCV(pipeline, tree_param_grid, scoring='r2', cv=3, n_jobs=-1)
        search.fit(train_df[predictors], train_df[target])
        
        best_params = search.best_params_
        # print(f"Year {current_oos_year} - Tree Best Params: {best_params}")

        # Re-fit on combined train_df + val_df using best params
        final_pipeline = Pipeline([
            ('model', DecisionTreeRegressor(random_state=42, 
                                         max_depth=best_params['model__max_depth'], 
                                         min_samples_leaf=best_params['model__min_samples_leaf']))
        ])
        
        combined_train_val_df = pd.concat([train_df, val_df])
        final_pipeline.fit(combined_train_val_df[predictors], combined_train_val_df[target])

        test_df.loc[:, 'tree_pred'] = final_pipeline.predict(test_df[predictors])
        
        oos_r2_year = r2_score(test_df[target], test_df['tree_pred'])
        all_oos_r2.append(oos_r2_year)
        print(f"Year {current_oos_year} - Decision Tree OOS R2: {oos_r2_year:.4f}")

        all_oos_predictions.append(test_df[['date', 'permno', 'tree_pred', target]])

        if i == len(list(next_window(df))) - 1: 
            joblib.dump(final_pipeline, 'outputs/tree_model_last_window.pkl')
            print(f"Saved final Decision Tree model for last window to outputs/tree_model_last_window.pkl")
            params_to_save = {
                'best_params_last_window': best_params,
                'oos_r2_last_window': oos_r2_year
            }
            with open('outputs/tree_params_last_window.json', 'w') as f:
                json.dump(params_to_save, f)
            print("Saved params for Decision Tree (last window) to outputs/tree_params_last_window.json")
            
    print("Finished processing all windows for Decision Tree model.")

    if not all_oos_predictions:
        print("No Decision Tree predictions were generated. Exiting.")
        return

    all_predictions_df = pd.concat(all_oos_predictions)
    overall_oos_r2 = np.mean(all_oos_r2) if all_oos_r2 else np.nan
    print(f'Overall Average OOS R² score for Decision Tree: {overall_oos_r2:.4f}')

    metrics_df = pd.DataFrame({'model': ['tree'], 'oos_r2': [overall_oos_r2]})
    metrics_file = 'outputs/metrics.csv'
    if os.path.exists(metrics_file):
        existing_metrics = pd.read_csv(metrics_file)
        existing_metrics = existing_metrics[existing_metrics['model'] != 'tree']
        metrics_df = pd.concat([existing_metrics, metrics_df])
    metrics_df.to_csv(metrics_file, index=False)
    print(f'Overall Decision Tree metrics saved to {metrics_file}')

    all_predictions_df = all_predictions_df.rename(columns={target: 'stock_exret'})
    tree_overall_port_ret = form_portfolio(all_predictions_df, 'tree_pred', n_stocks=50)
    tree_overall_port_ret.columns = ['tree_port_ret'] 
    tree_overall_port_ret.to_csv('outputs/tree_overall_port_ret.csv')
    print('Overall Decision Tree monthly portfolio returns saved to outputs/tree_overall_port_ret.csv')

    spy_returns = pd.read_excel('Data/SPY returns.xlsx', index_col=0)
    spy_returns.index = pd.to_datetime(spy_returns.index)
    perf_summary_df = calculate_performance_metrics(tree_overall_port_ret, spy_returns, 'tree')
    summary_file = 'outputs/perf_summary.csv'
    if os.path.exists(summary_file):
        existing_summary = pd.read_csv(summary_file)
        existing_summary = existing_summary[existing_summary['model'] != 'tree']
        perf_summary_df = pd.concat([existing_summary, perf_summary_df])
    perf_summary_df.to_csv(summary_file, index=False)
    print(f'Overall Decision Tree performance summary saved to {summary_file}')

if __name__ == "__main__":
    run_tree_model() 