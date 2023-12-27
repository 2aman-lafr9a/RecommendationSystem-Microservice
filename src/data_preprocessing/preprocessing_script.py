import pandas as pd
import numpy as np

file_path = 'players_football_ds.csv'
df = pd.read_csv(file_path)
# df.head()

# Describing the ds
print("\n--------- Describing the ds ---------")
print(df.describe())

# Discovering the ds
print("\n--------- Discovering the ds ---------")
print(df.info())

## Checking if there are any mssing values
print("\n--------- Messing values ---------")
print(df.isnull().sum())

# Check for duplicate rows
print("\n--------- Duplicating values or rows ---------")
print("Duplicate rows:", df.duplicated().sum())

# Droping any duplicate rows if existe
df = df.drop_duplicates()

# Droping rows that contains missing values
df = df.dropna()

print("Missing values after cleaning:", df.isnull().sum())
print("Duplicate rows after cleaning:", df.duplicated().sum())

from sklearn.preprocessing import StandardScaler


# Here just grouping the ages
bins = [15, 20, 25, 30, 35, 40, 50, 60]
labels = ['15-20', '21-25', '26-30', '31-35', '36-40', '41-50', '51-60']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Overall Rating Grouping
bins = [40, 50, 60, 70, 80, 90, 100]
labels = ['40-50', '51-60', '61-70', '71-80', '81-90', '91-100']
df['Overall_Group'] = pd.cut(df['Overall'], bins=bins, labels=labels, right=False)

# Log Transformation for Value(£)
df['Log_Value'] = np.log1p(df['Value(£)'])

# before log transformation
# original_skewness = df['Value(£)'].skew()

# after log transformation
# log_transformed_skewness = df['Log_Value'].skew()

# print(f"asymtre before log transformation: {original_skewness}")
# print(f"asymetre after log transformation: {log_transformed_skewness}")


# Combine Age and Overall Rating
df['Age_Rating'] = df['Age'] * df['Overall']

# Remove teh nationality
df = df.drop(['Nationality'], axis=1)


print(df.head())



