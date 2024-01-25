import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os


# Set the working directory to the location of your script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Construct the file path using os.path.join
file_path = os.path.join(script_dir, "../../data/processed/preprocessed_data_v1.csv")
df_players = pd.read_csv(file_path)


print(df_players.head())

features_for_clustering = ['Value(£)', 'Overall']

# Standardize the features
X = df_players[features_for_clustering]
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Elbow Method to find optimal k
inertia = []
possible_k_values = range(1, 11)

for k in possible_k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_standardized)
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(10, 6))
plt.plot(possible_k_values, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Choose the optimal value of k based on the elbow method analysis
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df_players['Cluster_Labels'] = kmeans.fit_predict(X_standardized)

# Visualize clusters using scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df_players['Overall'], df_players['Value(£)'], c=df_players['Cluster_Labels'], cmap='viridis')
plt.title('K-Means Clustering Based on Value(£) and Overall')
plt.xlabel('Overall')
plt.ylabel('Value(£)')
plt.show()

print("DataFrame with Cluster Assignment:")
print(df_players[['Name', 'Value(£)', 'Overall', 'Cluster_Labels']])

# Add a unique and incremented playerId column
df_players['playerId'] = range(1, len(df_players) + 1)

# Save the updated players dataset
df_players.to_csv("../../data/trained/clustered_players.csv", index=False)
pd.to_pickle(scaler, "../model/scaler_model.pkl")
pd.to_pickle(kmeans, "../model/kmeans_model.pkl")

