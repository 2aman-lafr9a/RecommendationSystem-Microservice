import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score, silhouette_score
import os

# from ...connection.database_connection import connect_to_database, extract_ratings_data, extract_insurances_data, close_connection

# Set the working directory to the location of your script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Construct the file path using os.path.join
file_path = os.path.join(script_dir, "preprocessed_data_v1.csv")
df_players = pd.read_csv(file_path)

# Connect to the PostgreSQL databases with different ports
# ratings_conn = connect_to_database('aman.francecentral.cloudapp.azure.com', 5433, 'postgres', 'postgres', 'rating_management')
# insurances_conn = connect_to_database('aman.francecentral.cloudapp.azure.com', 5432, 'postgres', 'postgres', 'agency_offers_database')

# Extract data from the ratings and insurances tables
# df_ratings = extract_ratings_data(ratings_conn)

# print(df_ratings)


# df_insurances = extract_insurances_data(insurances_conn)

# Close the database connections
# close_connection(ratings_conn)
# close_connection(insurances_conn)

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

# Compute Davies-Bouldin Index
# davies_bouldin = davies_bouldin_score(X_standardized, df_players['Cluster_Labels'])
# print(f"Davies-Bouldin Index: {davies_bouldin}")

# # Compute Silhouette Score
# silhouette_avg = silhouette_score(X_standardized, df_players['Cluster_Labels'])
# print(f"Silhouette Score: {silhouette_avg}")

# Compute cluster characteristics
# cluster_counts = df_players['Cluster_Labels'].value_counts().sort_index()
# cluster_means = df_players.groupby('Cluster_Labels').mean()[['Value(£)', 'Overall']]

# Visualize cluster characteristics
# plt.figure(figsize=(12, 8))
# plt.subplot(2, 1, 1)
# sns.barplot(x=cluster_counts.index, y=cluster_counts)
# plt.title('Number of Rows in Each Cluster')

# plt.subplot(2, 1, 2)
# sns.barplot(x=cluster_means.index, y=cluster_means['Overall'])
# plt.title('Average Overall Rating for Each Cluster')

# plt.tight_layout()
# plt.show()

# print("Cluster Characteristics (Mean Values):")
# print(cluster_means)

print("DataFrame with Cluster Assignment:")
print(df_players[['Name', 'Value(£)', 'Overall', 'Cluster_Labels']])

# Add a unique and incremented playerId column
df_players['playerId'] = range(1, len(df_players) + 1)

# Save the updated players dataset
df_players.to_csv("clustered_players.csv", index=False)
pd.to_pickle(scaler, "scaler_model.pkl")
pd.to_pickle(kmeans, "kmeans_model.pkl")

