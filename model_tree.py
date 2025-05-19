import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import joblib
import json
import os
from utils import next_window

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
    ('model', DecisionTreeRegressor(random_state=42))
])
grid = {
    'model__max_depth': [3, 5, 7, 10],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
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
with open('outputs/tree_params.json', 'w') as f:
    json.dump(result, f)
print('Best params and score saved to outputs/tree_params.json')

# Task 13: Re-fit on combined train+val using best params
best_params = search.best_params_
pipeline.set_params(**best_params)
pipeline.fit(pd.concat([train, val])[predictors], pd.concat([train, val])['stock_exret'])

# Save fitted model
joblib.dump(pipeline, 'outputs/tree_model.pkl')
print('Fitted model saved to outputs/tree_model.pkl')

# Test: reload model
loaded_model = joblib.load('outputs/tree_model.pkl')
print('Model reloaded successfully')

# Task 14: Generate predictions for stock_exret on test window
test = test.copy()  # Create a copy to avoid fragmentation
test.loc[:, 'tree_pred'] = loaded_model.predict(test[predictors])
print('Predictions saved to test DataFrame as tree_pred')
print('All predictions non-null:', test['tree_pred'].notna().all())

# Debug prints
print('\nTest DataFrame info:')
print('Number of rows:', len(test))
print('Date range:', test['date'].min(), 'to', test['date'].max())
print('Number of unique dates:', test['date'].nunique())
print('Sample of predictions:')
print(test[['date', 'tree_pred']].head())

# Task 15: Compute out-of-sample R² score
oos_r2 = r2_score(test['stock_exret'], test['tree_pred'])
print('Out-of-sample R² score:', oos_r2)

# Append to metrics.csv
metrics = pd.DataFrame({'model': ['tree'], 'oos_r2': [oos_r2]})
metrics.to_csv('outputs/metrics.csv', mode='a', header=not os.path.exists('outputs/metrics.csv'), index=False)
print('Metrics appended to outputs/metrics.csv')

# Task 16: Form monthly portfolio
test = test.copy()  # Create a copy to avoid fragmentation
test.loc[:, 'rank'] = test.groupby('date')['tree_pred'].rank(ascending=False)
top_50 = test[test['rank'] <= 50].copy()  # Create a copy to avoid SettingWithCopyWarning
top_50.loc[:, 'weight'] = 1 / 50

# Calculate portfolio returns properly
tree_port_ret = top_50.groupby('date').apply(
    lambda x: (x['stock_exret'] * x['weight']).sum()
).reset_index()
tree_port_ret.columns = ['date', 'tree_port_ret']
tree_port_ret.set_index('date', inplace=True)
tree_port_ret.to_csv('outputs/tree_port_ret.csv')
print('Monthly portfolio returns saved to outputs/tree_port_ret.csv')

# Task 17: Performance stats
# Read SPY returns
spy_returns = pd.read_excel('Data/SPY returns.xlsx', index_col=0)
spy_returns.index = pd.to_datetime(spy_returns.index)
# Align SPY returns to month end
def to_month_end(idx):
    return idx.to_period('M').to_timestamp('M')
spy_returns.index = to_month_end(spy_returns.index)
tree_port_ret = pd.read_csv('outputs/tree_port_ret.csv', index_col=0, parse_dates=True)
# Align portfolio return index to calendar month end
tree_port_ret.index = tree_port_ret.index.to_period('M').to_timestamp('M')

# Calculate performance metrics
alpha = tree_port_ret['tree_port_ret'].mean() - spy_returns['SPY_ret'].mean()
sharpe = (tree_port_ret['tree_port_ret'].mean() / tree_port_ret['tree_port_ret'].std()) * (12 ** 0.5)
stdev = tree_port_ret['tree_port_ret'].std() * (12 ** 0.5)
max_drawdown = (tree_port_ret['tree_port_ret'].cummax() - tree_port_ret['tree_port_ret']).max()
max_loss = tree_port_ret['tree_port_ret'].min()

# Append to perf_summary.csv
perf_summary = pd.DataFrame({
    'model': ['tree'],
    'alpha': [alpha],
    'sharpe': [sharpe],
    'stdev': [stdev],
    'max_drawdown': [max_drawdown],
    'max_loss': [max_loss]
})
perf_summary.to_csv('outputs/perf_summary.csv', mode='a', header=not os.path.exists('outputs/perf_summary.csv'), index=False)
print('Performance summary appended to outputs/perf_summary.csv') 