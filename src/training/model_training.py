import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def train_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/processed/preprocessed_data_v1.csv")

    # Read the preprocessed data
    df_players = pd.read_csv(file_path)

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
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../images/elbow_curve.png"))
    plt.close()

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
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../images/clustering_scatter_plot.png"))
    plt.close()

    print("DataFrame with Cluster Assignment:")
    print(df_players[['Name', 'Value(£)', 'Overall', 'Cluster_Labels']])

    # Add a unique and incremented playerId column
    df_players['playerId'] = range(1, len(df_players) + 1)

    # Save the updated players dataset
    df_players.to_csv("../../data/trained/clustered_players.csv", index=False)
    pd.to_pickle(scaler, "../model/scaler_model.pkl")
    pd.to_pickle(kmeans, "../model/kmeans_model.pkl")

if __name__ == "__main__":
    train_model()
