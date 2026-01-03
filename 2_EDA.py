import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv('../data/employee_data_cleaned.csv')

# Attrition count
sns.countplot(x='Attrition', data=df)
plt.title("Attrition Count")
plt.show()

# Average Monthly Income by Department
plt.figure(figsize=(8,5))
sns.barplot(x='Department', y='MonthlyIncome', data=df)
plt.title("Monthly Income by Department")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()
