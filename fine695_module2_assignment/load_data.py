import pandas as pd
import os
from utils import next_window

# Load the data
df = pd.read_csv('Data/homework_sample_big.csv')

# Load the list of predictor columns
predictors = pd.read_csv('Data/factors_char_list.csv')['variable'].tolist()

# Convert date to datetime and sort
df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
df = df.sort_values(['date', 'permno'])

# Task 07: Define time splits
train_end = pd.to_datetime('2014-12-31')
val_end = pd.to_datetime('2016-12-31')
test_start = pd.to_datetime('2017-01-01')
test_end = pd.to_datetime('2017-12-31')

def flag_set(row):
    if row['date'] <= train_end:
        return 'train'
    elif row['date'] <= val_end:
        return 'val'
    elif test_start <= row['date'] <= test_end:
        return 'test'
    else:
        return 'other'

df['set'] = df.apply(flag_set, axis=1)

# Create data overview for only the relevant columns
data_overview = pd.DataFrame({
    'dtype': df[predictors].dtypes,
    'null_count': df[predictors].isnull().sum()
})

# Ensure outputs directory exists
os.makedirs('outputs', exist_ok=True)

# Save summary to CSV
data_overview.to_csv('outputs/00_data_overview.csv')

# Print verification
print(f"DataFrame loaded with {len(df)} rows")
print(f"Data overview saved to outputs/00_data_overview.csv")
print(f"Number of predictor columns: {len(predictors)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Is date monotonic increasing: {df['date'].is_monotonic_increasing}")

# Print set counts and test set min date for verification
print(df['set'].value_counts())
print("Test set min date:", df.query("set=='test'")['date'].min())

# Task 06: Print sample row pairs for the same permno to check lag alignment
permno_sample = df['permno'].iloc[0]
sample = df[df['permno'] == permno_sample].head(3)
print("\nSample rows for permno", permno_sample)
print(sample[['date'] + predictors[:3] + ['ret_eom']])

# Test next_window generator (Task 08)
print("\nTesting next_window generator (first 3 splits):")
for i, (train, val, test) in enumerate(next_window(df)):
    print(f"Split {i+1}: train={len(train)}, val={len(val)}, test={len(test)})")
    if i == 2:
        break 