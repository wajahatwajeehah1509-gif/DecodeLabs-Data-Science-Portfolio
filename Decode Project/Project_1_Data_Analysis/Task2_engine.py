import pandas as pd
import numpy as np

# 1. Load the data
df = pd.read_csv(r'C:\Users\Wajeehah Wajahat\Downloads\Decode Project\cleaned_dataset.csv')

# 2. Remove IDs that don't add predictive value
cols_to_drop = ['OrderID', 'CustomerID', 'TrackingNumber']
df = df.drop(columns=cols_to_drop)

# 3. Create new features
df['Feature_BaseValuation'] = df['Quantity'] * df['UnitPrice']
# Handle division by zero just in case
df['Feature_CartDensity'] = df['ItemsInCart'] / df['Quantity'].replace(0, 1)
df['Feature_LogUnitPrice'] = np.log1p(df['UnitPrice'])

# 4. Convert categories into dummy variables
# drop_first=True helps avoid multicollinearity in encoding
categories = ['Product', 'PaymentMethod', 'OrderStatus', 'ReferralSource', 'CouponCode']
df = pd.get_dummies(df, columns=categories, drop_first=True)

# 5. Remove Collinearity (Redundancy |ρ| > 0.80)
# We select only numeric columns for the correlation matrix
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr().abs()

# The rest of the logic stays the same
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Identify columns to drop based on correlation
to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
print(f"Dropping collinear features: {to_drop}")
df = df.drop(columns=to_drop)

# 6. Save the final processed file
df.to_csv('transformed_feature_space.csv', index=False)
print("\n--- TASK 2 COMPLETE ---")
print("Data saved as 'transformed_feature_space.csv'")