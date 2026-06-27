import pandas as pd

# 1. Load the data
df = pd.read_csv('transformed_feature_space.csv')
df['Date'] = pd.to_datetime(df['Date'])

# 2. Check if the data looks the way we expect
def check_data_quality(df):
    issues = []
    
    # Check if columns are the right format
    expected_types = {'Quantity': 'int64', 'TotalPrice': 'float64'}
    for col, expected_type in expected_types.items():
        if col in df.columns and df[col].dtype != expected_type:
            issues.append(f"Type mismatch: {col} should be {expected_type}, but got {df[col].dtype}")
    
    # Make sure we don't have negative prices
    if (df['TotalPrice'] < 0).any():
        issues.append("Data error: TotalPrice contains negative values.")
    
    return issues

# 3. Get features that existed before a specific date
def get_features_at_date(df, target_date):
    # Filter for data up to the target date
    past_data = df[df['Date'] <= target_date]
    
    if past_data.empty:
        return None
        
    # Return the most recent entry
    return past_data.sort_values('Date').iloc[-1]

# Run the validation
errors = check_data_quality(df)
if errors:
    print("--- Data Issues Found ---")
    for error in errors:
        print(error)
else:
    print("Data validation passed: Everything looks correct.")

# Example lookup
lookup_date = pd.to_datetime('2025-01-01')
features = get_features_at_date(df, lookup_date)

print(f"\nFeatures found for {lookup_date.date()}:")
if features is not None:
    print(features[['Quantity', 'TotalPrice']])
else:
    print("No data found for this date.")

# Force data types to match our requirements
df['Quantity'] = df['Quantity'].astype('int64')
df['TotalPrice'] = df['TotalPrice'].astype('float64')

# Re-run the validation to check if everything is fixed
errors = check_data_quality(df)

if errors:
    print("--- Still finding issues ---")
    for error in errors:
        print(error)
else:
    print("All checks passed: Data is ready to go.")