import pandas as pd

# Load dataset
df = pd.read_csv('../data/employee_data.csv')
print("Initial Data:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Encode categorical columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in ['Gender', 'Department', 'JobRole', 'OverTime', 'Attrition']:
    df[col] = le.fit_transform(df[col])

print("\nData after encoding:")
print(df.head())

# Save cleaned data
df.to_csv('../data/employee_data_cleaned.csv', index=False)
