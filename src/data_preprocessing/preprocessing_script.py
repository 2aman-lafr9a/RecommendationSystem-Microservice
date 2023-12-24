import pandas as pd

file_path = 'data/raw/players_football_ds.csv'
df = pd.read_csv(file_path)

print(df.head())