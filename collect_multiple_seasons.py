import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")

if not API_KEY:
    raise ValueError("FOOTBALL_DATA_API_KEY not found in .env")

BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}

def fetch_season(season):
    """Fetch finished PL matches for one season."""
    endpoint = f"{BASE_URL}/competitions/PL/matches"
    params = {"season": season, "status": "FINISHED"}

    print(f"  → Fetching PL {season}/{season+1}...")
    r = requests.get(endpoint, headers=HEADERS, params=params)

    if r.status_code == 429:
        print("  ⏳ Rate limit hit. Sleeping 60s...")
        time.sleep(60)
        return fetch_season(season)

    if r.status_code != 200:
        print(f"  ✗ Error {r.status_code}: {r.text}")
        return None

    return r.json()

def parse_matches(json_data, season):
    """Extract the fields we need."""
    rows = []
    for m in json_data.get("matches", []):
        hs = m["score"]["fullTime"]["home"]
        as_ = m["score"]["fullTime"]["away"]
        if hs is None or as_ is None:
            continue
        rows.append({
            "date": m["utcDate"],
            "season": season,
            "home_team": m["homeTeam"]["name"],
            "away_team": m["awayTeam"]["name"],
            "home_score": hs,
            "away_score": as_,
            "league": m["competition"]["name"]
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    seasons = [2020, 2021, 2022, 2023, 2024]
    all_dfs = []

    for s in seasons:
        data = fetch_season(s)
        if data:
            df = parse_matches(data, s)
            print(f"     ✓ {len(df)} matches collected")
            all_dfs.append(df)
        time.sleep(6)  # respect rate limit

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        os.makedirs("data/raw", exist_ok=True)
        combined.to_csv("data/raw/pl_multi_season.csv", index=False)

        print(f"\n✅ Total: {len(combined)} matches saved to data/raw/pl_multi_season.csv")
        print(combined.head())
