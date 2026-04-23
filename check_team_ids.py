import pandas as pd

df = pd.read_csv("data/cache_2023-24_Regular_Season.csv")
print(df[["TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME"]].drop_duplicates().sort_values("TEAM_ABBREVIATION").to_string())