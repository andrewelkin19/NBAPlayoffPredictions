import pandas as pd

#df = pd.read_csv("data/training_data.csv")
#print(df[df["SEASON"] == "2022-23"]["TEAM_ID"].unique())

# check home team win percentage in playoff data
#df = pd.read_csv("data/training_data.csv")
#print(df["LABEL"].mean())

from scrape_bpm import BBREF_TO_NBA_API

for season in ["2005-06", "2006-07", "2007-08"]:
    bpm = pd.read_csv(f"data/bpm/{season}_bpm.csv")
    reg = pd.read_csv(f"data/cache_{season}_Regular_Season.csv")
    
    nba_abbrevs  = set(reg["TEAM_ABBREVIATION"].unique())
    bbref_abbrevs = set(bpm["TEAM_BBREF"].unique())
    
    # Which BBRef abbreviations don't map to any nba_api abbreviation?
    unmapped = {a for a in bbref_abbrevs if BBREF_TO_NBA_API.get(a) not in nba_abbrevs}
    print(f"{season}: unmapped BBRef abbrevs = {unmapped}")
    print(f"         nba_api abbrevs = {sorted(nba_abbrevs)}")