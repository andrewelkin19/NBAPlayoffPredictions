"""
NBA Playoff Outcome Prediction - BPM Scraper
CS830 Final Project - Andrew Elkin

Scrapes Box Plus/Minus (BPM) from Basketball Reference using the
basketball_reference_scraper library by vishaalagartha, which uses
headless Chrome (Selenium) to bypass bot detection.

Fetches the league-wide advanced stats page for each season — one
Selenium call per season (18 total) rather than one per team.

Requirements:
  pip install basketball_reference_scraper
  Chrome must be installed on your machine.

Usage:
  python scrape_bpm.py                  # scrape all 18 seasons
  python scrape_bpm.py --test           # test 2022-23 only
  python scrape_bpm.py --validate       # print top 10 BPM players per season

Output:
  data/bpm/{season}_bpm.csv             # one file per season
  data/bpm/all_seasons_bpm.csv          # combined across all seasons

After running this, regenerate training data with BPM columns:
  python DataScrape.py --clutch --star-players --bpm
"""

import os
import argparse
import pandas as pd

from basketball_reference_scraper.request_utils import get_selenium_wrapper

# ── Config ────────────────────────────────────────────────────────────────────

SEASONS = [
    "2005-06", "2006-07", "2007-08", "2008-09", "2009-10",
    "2010-11",            "2012-13", "2013-14", "2014-15",
    "2015-16", "2016-17", "2017-18", "2018-19",
                          "2020-21", "2021-22", "2022-23",
                          "2023-24", "2024-25"
]

DATA_DIR = os.path.join("data", "bpm")

# BBRef abbreviations → nba_api TEAM_ABBREVIATION
BBREF_TO_NBA_API = {
    "ATL": "ATL", "BOS": "BOS", "BRK": "BKN", "CHO": "CHA", "CHI": "CHI",
    "CLE": "CLE", "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GSW": "GSW",
    "HOU": "HOU", "IND": "IND", "LAC": "LAC", "LAL": "LAL", "MEM": "MEM",
    "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NOP": "NOP", "NYK": "NYK",
    "OKC": "OKC", "ORL": "ORL", "PHI": "PHI", "PHO": "PHX", "POR": "POR",
    "SAC": "SAC", "SAS": "SAS", "TOR": "TOR", "UTA": "UTA", "WAS": "WAS",
    "NJN": "NJN", "SEA": "SEA", "NOH": "NOH", "NOK": "NOK",
    "CHA": "CHA", "CHH": "CHA",
}


# ── Scraping ──────────────────────────────────────────────────────────────────

def season_to_year(season: str) -> int:
    """'2022-23' → 2023"""
    return int(season[:4]) + 1


def scrape_bpm_season(season: str) -> pd.DataFrame | None:
    """
    Scrape BPM for all players in a given season using headless Chrome.

    Parses via BeautifulSoup data-stat attributes rather than pd.read_html,
    which is fragile with BBRef's repeated-header-row table structure.
    """
    cache_path = os.path.join(DATA_DIR, f"{season}_bpm.csv")

    if os.path.exists(cache_path):
        print(f"  Loading from cache: {cache_path}")
        return pd.read_csv(cache_path)

    year  = season_to_year(season)
    url   = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
    xpath = '//table[@id="advanced"]'

    print(f"  Fetching via Selenium: {url}")
    table_html = get_selenium_wrapper(url, xpath)

    if not table_html:
        print(f"  [error] Selenium returned None — table not found or page error")
        return None

    print(f"  Table HTML received ({len(table_html)} chars) — parsing...")

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(table_html, "html.parser")

    rows = []
    for tr in soup.find_all("tr"):
        # Skip header rows — they contain <th data-stat="ranker"> cells
        if tr.find(lambda t: t.name in ("th", "td") and
                   t.get("data-stat") == "ranker" and
                   t.get_text(strip=True) in ("Rk", "")):
            # Check if it's a real header row vs a data row with a ranker th
            if tr.find("th", {"data-stat": "name_display"}):
                continue  # it's a header row

        def get(stat):
            cell = tr.find(["td", "th"], {"data-stat": stat})
            return cell.get_text(strip=True) if cell else None

        player = get("name_display")
        team   = get("team_name_abbr")
        mp     = get("mp")
        bpm    = get("bpm")

        # Skip header rows and the League Average row
        if not player or player in ("Player", "League Average", ""):
            continue
        if not bpm or bpm == "":
            continue
        if team in ("2TM", "3TM", "4TM", "5TM"):
            continue   # skip multi-team aggregate rows, keep only per-team rows

        try:
            rows.append({
                "PLAYER":     player,
                "TEAM_BBREF": team or "",
                "MP":         float(mp)  if mp  else 0.0,
                "BPM":        float(bpm),
                "SEASON":     season,
            })
        except ValueError:
            continue

    if not rows:
        print(f"  [error] Parsed 0 rows — HTML may be malformed or table empty")
        return None

    print(f"  Parsed {len(rows)} player rows")
    df = pd.DataFrame(rows)

    # For traded players: keep only TOT row (season totals across teams)
    multi_team  = df[df["TEAM_BBREF"] == "TOT"]["PLAYER"].unique()
    single_rows = df[~df["PLAYER"].isin(multi_team)]
    tot_rows    = df[df["TEAM_BBREF"] == "TOT"]
    df = pd.concat([single_rows, tot_rows], ignore_index=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"  ✓ {len(df)} players saved to {cache_path}")
    return df


# ── Validation ────────────────────────────────────────────────────────────────

def validate_season(df: pd.DataFrame, season: str):
    top10 = df.nlargest(10, "BPM")[["PLAYER", "TEAM_BBREF", "BPM", "MP"]]
    print(f"\n  Top 10 BPM — {season}:")
    for _, row in top10.iterrows():
        print(f"    {row['PLAYER']:<28} {row['TEAM_BBREF']:>4}  "
              f"BPM={row['BPM']:>+6.1f}  MP={row['MP']:>5.0f}")


# ── Integration helpers (used by DataScrape.py) ───────────────────────────────

def load_bpm_season(season: str) -> pd.DataFrame | None:
    """Load cached BPM. Returns None if scrape_bpm.py hasn't been run yet."""
    path = os.path.join(DATA_DIR, f"{season}_bpm.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def compute_star_bpm_features(
    team_features: pd.DataFrame,
    bpm_df: pd.DataFrame,
    min_mp: float = 1000.0,
) -> pd.DataFrame:
    """
    Adds BPM-based star player columns to team_features:

      TOP1_BPM      — best player's BPM on the roster
      TOP2_AVG_BPM  — average BPM of top 2 qualified players
      TOP3_AVG_BPM  — average BPM of top 3 qualified players

    min_mp=1000 minutes ≈ 12+ mpg over 82 games.
    Teams with no qualifying players receive 0.0 (league average BPM).
    """
    qualified = bpm_df[bpm_df["MP"] >= min_mp].copy()
    qualified["NBA_API_ABBREV"] = qualified["TEAM_BBREF"].map(BBREF_TO_NBA_API)

    star_rows = []
    for _, team_row in team_features.iterrows():
        abbrev = str(team_row.get("TEAM_ABBREVIATION", ""))
        team_players = (
            qualified[qualified["NBA_API_ABBREV"] == abbrev]
            .sort_values("BPM", ascending=False)
        )

        if len(team_players) == 0:
            top1 = top2 = top3 = 0.0
        elif len(team_players) == 1:
            top1 = top2 = top3 = float(team_players.iloc[0]["BPM"])
        elif len(team_players) == 2:
            top1 = float(team_players.iloc[0]["BPM"])
            top2 = float(team_players.head(2)["BPM"].mean())
            top3 = top2
        else:
            top1 = float(team_players.iloc[0]["BPM"])
            top2 = float(team_players.head(2)["BPM"].mean())
            top3 = float(team_players.head(3)["BPM"].mean())

        star_rows.append({
            "TEAM_ID":      int(team_row["TEAM_ID"]),
            "TOP1_BPM":     top1,
            "TOP2_AVG_BPM": top2,
            "TOP3_AVG_BPM": top3,
        })

    star_df = pd.DataFrame(star_rows)
    merged  = team_features.merge(star_df, on="TEAM_ID", how="left")
    for col in ["TOP1_BPM", "TOP2_AVG_BPM", "TOP3_AVG_BPM"]:
        merged[col] = merged[col].fillna(0.0)

    n_zero = (merged["TOP1_BPM"] == 0.0).sum()
    if n_zero > 0:
        print(f"  [warning] {n_zero} teams have TOP1_BPM=0.0 — "
              f"check BBREF_TO_NBA_API mapping")

    return merged


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scrape BPM from Basketball Reference via headless Chrome"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test with 2022-23 only"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Print top 10 BPM players per season"
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    seasons = ["2022-23"] if args.test else SEASONS

    print(f"Scraping BPM for {len(seasons)} season(s) via headless Chrome")
    print(f"Output: {DATA_DIR}\n")

    all_dfs = []
    for season in seasons:
        print(f"Season: {season}")
        df = scrape_bpm_season(season)
        if df is not None:
            if args.validate:
                validate_season(df, season)
            all_dfs.append(df)
        else:
            print(f"  [skip] No data for {season}")
        print()

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out_path = os.path.join(DATA_DIR, "all_seasons_bpm.csv")
        combined.to_csv(out_path, index=False)
        print(f"✓ {out_path}  ({len(combined)} rows, "
              f"{combined['SEASON'].nunique()} seasons)")
    else:
        print("No data scraped.")


if __name__ == "__main__":
    main()