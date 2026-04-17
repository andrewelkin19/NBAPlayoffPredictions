import pandas as pd

df = pd.read_csv("data/training_data.csv")
print(df.shape)
print(df["LABEL"].value_counts())
print(df.isnull().sum())
print(df["SEASON"].value_counts().sort_index())