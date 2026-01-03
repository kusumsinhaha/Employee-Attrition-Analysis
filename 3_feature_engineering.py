import pandas as pd

# Load cleaned data
df = pd.read_csv('../data/employee_data_cleaned.csv')

# Example: Create new feature - Years since last promotion (simulated)
import numpy as np
df['YearsSincePromotion'] = np.random.randint(0, df['YearsAtCompany']+1, size=df.shape[0])

# Feature and target selection
X = df.drop(['EmployeeID','Attrition'], axis=1)
y = df['Attrition']

print("Features:")
print(X.head())
print("\nTarget:")
print(y.head())
