import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# Load your API key from .env
load_dotenv()
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")

if not API_KEY:
    raise ValueError("FOOTBALL_DATA_API_KEY not found in .env file")

# API configuration
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}

def fetch_premier_league_matches(season=2023):
    """
    Fetch finished Premier League matches for a given season.
    Season is the starting year (2023 = 2023/24 season).
    """
    endpoint = f"{BASE_URL}/competitions/PL/matches"
    params = {
        "season": season,
        "status": "FINISHED"
    }

    print(f"Fetching PL {season}/{season+1} matches...")
    response = requests.get(endpoint, headers=HEADERS, params=params)

    # Check for rate limiting or errors
    if response.status_code == 429:
        print("Rate limit hit. Waiting 60 seconds...")
        time.sleep(60)
        return None

    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return None

    return response.json()

def parse_matches(json_data):
    """Extract the fields we need from the API response."""
    matches = []

    for match in json_data.get("matches", []):
        # Skip matches without scores
        home_score = match["score"]["fullTime"]["home"]
        away_score = match["score"]["fullTime"]["away"]

        if home_score is None or away_score is None:
            continue

        matches.append({
            "date": match["utcDate"],
            "home_team": match["homeTeam"]["name"],
            "away_team": match["awayTeam"]["name"],
            "home_score": home_score,
            "away_score": away_score,
            "league": match["competition"]["name"]
        })

    return pd.DataFrame(matches)

# Main execution
if __name__ == "__main__":
    data = fetch_premier_league_matches(season=2023)

    if data:
        df = parse_matches(data)
        print(f"Collected {len(df)} matches")

        # Save to CSV
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv("data/raw/pl_2023.csv", index=False)
        print("Saved to data/raw/pl_2023.csv")

        # Preview
        print("\nFirst 5 rows:")
        print(df.head())
