# %% [markdown]
# ## **Part 1-  Data Preproccessing**

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
file_path = os.path.join(script_dir, "../../data/raw/players_football_ds.csv")

df = pd.read_csv(file_path)
df.head()

# %% [markdown]
# ## **Step 1: Data Exploration**

# %% [markdown]
# + **LOGIC**

# %%
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

# %% [markdown]
# + **INTERPRETATION**

# %% [markdown]
# ## **Step 2: Data Cleaning and Handling Missing Values**

# %% [markdown]
# + **LOGIC**

# %%
# Droping any duplicate rows if existe
df = df.drop_duplicates()

# Droping rows that contains missing values
df = df.dropna()

print("Missing values after cleaning:", df.isnull().sum())
print("Duplicate rows after cleaning:", df.duplicated().sum())

# %% [markdown]
# + **INTERPRETATION**

# %% [markdown]
# ## **Step 3: Feature Engineering**

# %% [markdown]
# + **LOGIC**

# %%

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

# %%

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df['Value(£)'], kde=True)
plt.title('Original Distribution')

plt.subplot(1, 2, 2)
sns.histplot(df['Log_Value'], kde=True)
plt.title('Log-Transformed Distribution')

plt.show()

# %% [markdown]
# + **INTERPRETATION**

# %% [markdown]
# ## **Step 4: Data Normalization/Scaling (if needed)**

# %% [markdown]
# + **LOGIC**

# %%
print("-----------BEFORE----------")
print(df.head())


# extract the features for normalization
features_to_normalize = ['Age', 'Overall', 'Value(£)', 'Log_Value', 'Age_Rating']

scaler = StandardScaler()

# normalize the selected features
df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])


print("\n-----------AFTER----------")
print(df.head())


# %% [markdown]
# + **INTERPRETATION**

# %% [markdown]
# ## **Step 5: Save Preprocessed Data**

# %% [markdown]
# + **LOGIC**

# %%
# specify dest path
output_path = '../../data/processed/preprocessed_data_v1.csv'

# Save the preprocessed df in new csv file
df.to_csv(output_path, index=False)

print(f"Preprocessed data saved to: {output_path}")

# df_processed = pd.read_csv("../../data/processed/preprocessed_data.csv")
# df_processed.head()

# %% [markdown]
# + **INTERPRETATION**


