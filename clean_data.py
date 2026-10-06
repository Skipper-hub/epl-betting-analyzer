import os
import pandas as pd
import numpy as np

TEAM_NAME_MAP = {
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Nott'm Forest": "Nottingham Forest",
    "Newcastle": "Newcastle United",
    "Tottenham": "Tottenham Hotspur",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
    "Sheffield United": "Sheffield United",
    "Luton": "Luton Town",
    "Ipswich": "Ipswich Town",
    "West Brom": "West Bromwich Albion",
    "Norwich": "Norwich City",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Brighton": "Brighton & Hove Albion",
    "Burnley": "Burnley",
    "Bournemouth": "AFC Bournemouth",
    "Watford": "Watford",
}

def load_raw(path="data/raw/pl_fd_uk.csv"):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} raw matches")
    return df

def clean(df):
    df["date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df["home_team"] = df["HomeTeam"].replace(TEAM_NAME_MAP)
    df["away_team"] = df["AwayTeam"].replace(TEAM_NAME_MAP)
    df["home_score"] = df["FTHG"]
    df["away_score"] = df["FTAG"]

    before = len(df)
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"])
    print(f"Dropped {before - len(df)} invalid/duplicate rows")

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    df["total_goals"] = df["home_score"] + df["away_score"]
    df["goal_diff"] = df["home_score"] - df["away_score"]
    df["result"] = "D"
    df.loc[df["goal_diff"] > 0, "result"] = "H"
    df.loc[df["goal_diff"] < 0, "result"] = "A"
    df["btts"] = ((df["home_score"] > 0) & (df["away_score"] > 0)).astype(int)
    df["over_2_5"] = (df["total_goals"] > 2.5).astype(int)

    # 1X2 odds
    df["odds_home"] = pd.to_numeric(df.get("B365H"), errors="coerce")
    df["odds_draw"] = pd.to_numeric(df.get("B365D"), errors="coerce")
    df["odds_away"] = pd.to_numeric(df.get("B365A"), errors="coerce")

    # ★ NEW: Over/Under 2.5 odds (B365>2.5 = over, B365<2.5 = under)
    df["odds_over_2_5"]  = pd.to_numeric(df.get("B365>2.5"), errors="coerce")
    df["odds_under_2_5"] = pd.to_numeric(df.get("B365<2.5"), errors="coerce")

    # Match stats
    df["home_shots"] = pd.to_numeric(df.get("HS"), errors="coerce")
    df["away_shots"] = pd.to_numeric(df.get("AS"), errors="coerce")
    df["home_shots_on_target"] = pd.to_numeric(df.get("HST"), errors="coerce")
    df["away_shots_on_target"] = pd.to_numeric(df.get("AST"), errors="coerce")
    df["home_corners"] = pd.to_numeric(df.get("HC"), errors="coerce")
    df["away_corners"] = pd.to_numeric(df.get("AC"), errors="coerce")

    keep = [
        "date", "season_label", "home_team", "away_team",
        "home_score", "away_score", "total_goals", "goal_diff",
        "result", "btts", "over_2_5",
        "odds_home", "odds_draw", "odds_away",
        "odds_over_2_5", "odds_under_2_5",       # ★ NEW
        "home_shots", "away_shots",
        "home_shots_on_target", "away_shots_on_target",
        "home_corners", "away_corners",
    ]
    df = df[keep].sort_values("date").reset_index(drop=True)
    return df

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    df = load_raw()
    df = clean(df)

    out = "data/processed/pl_clean.csv"
    df.to_csv(out, index=False)

    print(f"\n✅ Cleaned: {len(df)} matches → {out}")
    print("\n--- Odds coverage ---")
    for c in ["odds_home", "odds_draw", "odds_away",
              "odds_over_2_5", "odds_under_2_5"]:
        print(f"  {c}: {df[c].notna().sum()}/{len(df)}")
    print("\n--- Sample rows with over/under odds ---")
    print(df[["date", "home_team", "away_team", "total_goals",
              "over_2_5", "odds_over_2_5", "odds_under_2_5"]].head())
