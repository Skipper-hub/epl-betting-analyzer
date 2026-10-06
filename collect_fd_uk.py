import os
import time
import requests
import pandas as pd
from io import StringIO

SEASON_MAP = {
    "2021": "2020/21",
    "2122": "2021/22",
    "2223": "2022/23",
    "2324": "2023/24",
    "2425": "2024/25",
}

BASE = "https://www.football-data.co.uk/mmz4281"

def fetch_season(code):
    url = f"{BASE}/{code}/E0.csv"
    print(f"  → Downloading {SEASON_MAP.get(code, code)} ({url})")
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        print(f"     ✗ Failed: {r.status_code}")
        return None
    return pd.read_csv(StringIO(r.text))

def main():
    frames = []
    for code, label in SEASON_MAP.items():
        df = fetch_season(code)
        if df is None or df.empty:
            continue
        df["season_label"] = label
        frames.append(df)
        print(f"     ✓ {len(df)} matches")
        time.sleep(1)

    if not frames:
        print("No data collected.")
        return

    combined = pd.concat(frames, ignore_index=True)
    os.makedirs("data/raw", exist_ok=True)
    out = "data/raw/pl_fd_uk.csv"
    combined.to_csv(out, index=False)
    print(f"\n✅ {len(combined)} matches saved → {out}")
    print("\nColumns:", len(combined.columns))
    print(list(combined.columns)[:25], "...")

if __name__ == "__main__":
    main()
