import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from utils import next_window

# Load data
df = pd.read_csv('Data/homework_sample_big.csv')
df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
df = df.sort_values(['date', 'permno'])
predictors = pd.read_csv('Data/factors_char_list.csv')['variable'].tolist()

# Get first window's train split
train, val, test = next_window(df).__next__()

# Median imputation (Task 10)
medians = train[predictors].median()
train[predictors] = train[predictors].fillna(medians)
val[predictors] = val[predictors].fillna(medians)
test[predictors] = test[predictors].fillna(medians)

# Check for NaNs
print('NaNs in train:', train[predictors].isna().sum().sum())
print('NaNs in val:', val[predictors].isna().sum().sum())
print('NaNs in test:', test[predictors].isna().sum().sum())

# Fit scaler on training predictors
scaler = StandardScaler()
scaler.fit(train[predictors])

# Ensure outputs directory exists
os.makedirs('outputs', exist_ok=True)

# Save scaler
joblib.dump(scaler, 'outputs/scaler.pkl')
print('Scaler saved to outputs/scaler.pkl')

# Test: reload scaler and check mean
scaler2 = joblib.load('outputs/scaler.pkl')
train_scaled = scaler2.transform(train[predictors])
print('Mean of scaled train predictors (should be ~0):', train_scaled.mean()) 