import pandas as pd

#df = pd.read_csv("data/training_data.csv")
#print(df[df["SEASON"] == "2022-23"]["TEAM_ID"].unique())

# check home team win percentage in playoff data
df = pd.read_csv("data/training_data.csv")
print(df["LABEL"].mean())