import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import joblib
import json
import os
from utils import next_window
from portfolio_utils import form_portfolio, calculate_performance_metrics

# Load data and predictors
df = pd.read_csv('Data/homework_sample_big.csv')
df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
df = df.sort_values(['date', 'permno'])
predictors = pd.read_csv('Data/factors_char_list.csv')['variable'].tolist()

# Get first window's train/val/test splits
train, val, test = next_window(df).__next__()

# Median imputation (as in preprocess.py)
medians = train[predictors].median()
train[predictors] = train[predictors].fillna(medians)
val[predictors] = val[predictors].fillna(medians)
test[predictors] = test[predictors].fillna(medians)

# Pipeline and grid
pipeline = Pipeline([
    ('model', MLPRegressor(random_state=42, max_iter=1000))
])
grid = {
    'model__hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'model__activation': ['relu', 'tanh'],
    'model__alpha': [0.0001, 0.001, 0.01]
}

# GridSearchCV on train, score=r2
search = GridSearchCV(pipeline, grid, scoring='r2', cv=3, n_jobs=-1)
search.fit(train[predictors], train['stock_exret'])

# Evaluate on validation split
y_val_pred = search.predict(val[predictors])
best_score = r2_score(val['stock_exret'], y_val_pred)

# Save best params and score
os.makedirs('outputs', exist_ok=True)
result = {'best_params': search.best_params_, 'best_score': best_score}
with open('outputs/nn2_params.json', 'w') as f:
    json.dump(result, f)
print('Best params and score saved to outputs/nn2_params.json')

# Task 13: Re-fit on combined train+val using best params
best_params = search.best_params_
pipeline.set_params(**best_params)
pipeline.fit(pd.concat([train, val])[predictors], pd.concat([train, val])['stock_exret'])

# Save fitted model
joblib.dump(pipeline, 'outputs/nn2_model.pkl')
print('Fitted model saved to outputs/nn2_model.pkl')

# Test: reload model
loaded_model = joblib.load('outputs/nn2_model.pkl')
print('Model reloaded successfully')

# Task 14: Generate predictions for stock_exret on test window
test = test.copy()  # Create a copy to avoid fragmentation
test.loc[:, 'nn2_pred'] = loaded_model.predict(test[predictors])
print('Predictions saved to test DataFrame as nn2_pred')
print('All predictions non-null:', test['nn2_pred'].notna().all())

# Task 15: Compute out-of-sample R² score
oos_r2 = r2_score(test['stock_exret'], test['nn2_pred'])
print('Out-of-sample R² score:', oos_r2)

# Append to metrics.csv
metrics = pd.DataFrame({'model': ['nn2'], 'oos_r2': [oos_r2]})
metrics.to_csv('outputs/metrics.csv', mode='a', header=not os.path.exists('outputs/metrics.csv'), index=False)
print('Metrics appended to outputs/metrics.csv')

# Task 16: Form monthly portfolio
# Form portfolio using utility function
nn2_port_ret = form_portfolio(test, 'nn2_pred', n_stocks=50)
nn2_port_ret.columns = ['nn2_port_ret']  # Rename to match model name
nn2_port_ret.to_csv('outputs/nn2_port_ret.csv')
print('Monthly portfolio returns saved to outputs/nn2_port_ret.csv')

# Task 17: Performance stats
# Read SPY returns
spy_returns = pd.read_excel('Data/SPY returns.xlsx', index_col=0)
spy_returns.index = pd.to_datetime(spy_returns.index)

# Calculate performance metrics using utility function
perf_summary = calculate_performance_metrics(
    nn2_port_ret,
    spy_returns,
    'nn2'
)

# Append to perf_summary.csv
perf_summary.to_csv('outputs/perf_summary.csv', mode='a', header=not os.path.exists('outputs/perf_summary.csv'), index=False)
print('Performance summary appended to outputs/perf_summary.csv') 