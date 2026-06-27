import pandas as pd

# 1. Load the dataset
file_path = r'C:\Users\Wajeehah Wajahat\Downloads\Decode Project\Dataset.csv'
df = pd.read_csv(file_path)

# 2. Check the size of the data
print("--- Checking Dataset Size ---")
print(f"Total Rows and Columns: {df.shape}\n")

# 3. Clean up any duplicate records
print("--- Removing Duplicates ---")
num_duplicates = df.duplicated().sum()
print(f"Found {num_duplicates} duplicates to remove.")
df = df.drop_duplicates()

# 4. Fill in missing Coupon Codes
# If there is no code, we'll label it as 'None' for clarity
df['CouponCode'] = df['CouponCode'].fillna('None')

# 5. Convert the 'Date' column to proper date format
# This makes it much easier to perform time-based analysis later
df['Date'] = pd.to_datetime(df['Date'])

# 6. Final verification
print("\n--- All Done! Here is your cleaned data structure ---")
df.info()