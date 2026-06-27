import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# 1. Load the dataset
file_path = r'C:\Users\Wajeehah Wajahat\Downloads\Decode Project\Dataset.csv'
df = pd.read_csv(file_path)

print(f"Starting with {df['CouponCode'].isnull().sum()} missing CouponCodes.")

# 2. Advanced Imputation: Filling gaps using KNN
# We use existing data patterns to 'guess' the most likely coupon code for missing rows.

# First, convert text codes into numbers so the math works
unique_codes = df['CouponCode'].dropna().unique()
code_map = {code: i for i, code in enumerate(unique_codes)}
reverse_map = {i: code for i, code in enumerate(unique_codes)}

df['CouponCode_encoded'] = df['CouponCode'].map(code_map)

# Pick the columns that help define a user's behavior
features = ['Quantity', 'UnitPrice', 'ItemsInCart', 'TotalPrice', 'CouponCode_encoded']

# Use KNN (3 neighbors) to fill the gaps
imputer = KNNImputer(n_neighbors=3)
df[features] = imputer.fit_transform(df[features])

# Convert numbers back to the original coupon text
df['CouponCode_encoded'] = df['CouponCode_encoded'].round().astype(int)
df['CouponCode'] = df['CouponCode_encoded'].map(reverse_map)

# Remove the temporary helper column
df = df.drop(columns=['CouponCode_encoded'])

print(f"Imputation complete! Remaining missing codes: {df['CouponCode'].isnull().sum()}")

# 3. Smoothing out the data: Removing outliers
# We use the IQR method to 'clip' extreme values that might skew our analysis.
print("Cleaning up numerical outliers...")

cols_to_clean = ['Quantity', 'UnitPrice', 'TotalPrice']

for col in cols_to_clean:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    # Identify how many values were extreme
    outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"- Clipped {outlier_count} values in {col}")
    
    # Bring the extreme values back into a reasonable range
    df[col] = df[col].clip(lower=lower, upper=upper)

# 4. Save the result
df.to_csv('cleaned_dataset.csv', index=False)
print("Success! Your cleaned file is ready: 'cleaned_dataset.csv'")