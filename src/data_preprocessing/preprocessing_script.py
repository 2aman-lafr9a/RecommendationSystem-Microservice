import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

def data_preprocessing():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/raw/players_football_ds.csv")

    # Load the dataset
    df = pd.read_csv(file_path)

    # Specify the directory to save images
    images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../images")
    os.makedirs(images_dir, exist_ok=True)

    # Step 1: Data Exploration
    print("\n--------- Describing the ds ---------")
    print(df.describe())

    print("\n--------- Discovering the ds ---------")
    print(df.info())

    print("\n--------- Missing values ---------")
    print(df.isnull().sum())

    print("\n--------- Duplicating values or rows ---------")
    print("Duplicate rows:", df.duplicated().sum())

    # Step 2: Data Cleaning and Handling Missing Values
    df = df.drop_duplicates()
    df = df.dropna()

    print("Missing values after cleaning:", df.isnull().sum())
    print("Duplicate rows after cleaning:", df.duplicated().sum())

    plt.figure()
    sns.histplot(df['Value(£)'], kde=True)
    plt.title('Cleaned Distribution')
    plt.savefig(os.path.join(images_dir, "cleaned_distribution.png"))
    plt.close()

    # Step 3: Feature Engineering
    bins = [15, 20, 25, 30, 35, 40, 50, 60]
    labels = ['15-20', '21-25', '26-30', '31-35', '36-40', '41-50', '51-60']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    bins = [40, 50, 60, 70, 80, 90, 100]
    labels = ['40-50', '51-60', '61-70', '71-80', '81-90', '91-100']
    df['Overall_Group'] = pd.cut(df['Overall'], bins=bins, labels=labels, right=False)

    df['Log_Value'] = np.log1p(df['Value(£)'])
    df['Age_Rating'] = df['Age'] * df['Overall']
    df = df.drop(['Nationality'], axis=1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df['Value(£)'], kde=True)
    plt.title('Original Distribution')
    plt.savefig(os.path.join(images_dir, "original_distribution.png"))
    plt.close()

    plt.subplot(1, 2, 2)
    sns.histplot(df['Log_Value'], kde=True)
    plt.title('Log-Transformed Distribution')
    plt.savefig(os.path.join(images_dir, "log_transformed_distribution.png"))
    plt.close()

    # Step 4: Data Normalization/Scaling
    print("-----------BEFORE----------")
    print(df.head())

    features_to_normalize = ['Age', 'Overall', 'Value(£)', 'Log_Value', 'Age_Rating']
    scaler = StandardScaler()
    df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

    print("\n-----------AFTER----------")
    print(df.head())

    # Step 5: Save Preprocessed Data
    output_path = '../../data/processed/preprocessed_data_v1.csv'
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to: {output_path}")

if __name__ == "__main__":
    data_preprocessing()
