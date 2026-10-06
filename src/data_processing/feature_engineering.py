import pandas as pd
import numpy as np
from collections import defaultdict


class FeatureEngineer:
    def __init__(self, df, form_window=5, h2h_window=5):
        self.df = df.sort_values("date").reset_index(drop=True)
        self.form_window = form_window
        self.h2h_window = h2h_window
        self.team_history = defaultdict(list)
        self.h2h_history = defaultdict(list)
        self.last_match_date = {}

    def _team_points(self, match, team):
        if match["home_team"] == team:
            if match["result"] == "H": return 3
            if match["result"] == "D": return 1
            return 0
        else:
            if match["result"] == "A": return 3
            if match["result"] == "D": return 1
            return 0

    def _goals_for(self, match, team):
        return match["home_score"] if match["home_team"] == team else match["away_score"]

    def _goals_against(self, match, team):
        return match["away_score"] if match["home_team"] == team else match["home_score"]

    def _sot_for(self, match, team):
        col = "home_shots_on_target" if match["home_team"] == team else "away_shots_on_target"
        v = match.get(col)
        return v if pd.notna(v) else 0

    def _team_form(self, team):
        hist = self.team_history[team][-self.form_window:]
        if not hist:
            return {"avg_scored": 1.4, "avg_conceded": 1.4, "ppg": 1.3, "avg_sot": 4.5}
        n = len(hist)
        return {
            "avg_scored":   sum(self._goals_for(m, team) for m in hist) / n,
            "avg_conceded": sum(self._goals_against(m, team) for m in hist) / n,
            "ppg":          sum(self._team_points(m, team) for m in hist) / n,
            "avg_sot":      sum(self._sot_for(m, team) for m in hist) / n,
        }

    def _venue_form(self, team, venue):
        hist = [m for m in self.team_history[team][-15:]
                if (m["home_team"] == team and venue == "H")
                or (m["away_team"] == team and venue == "A")]
        hist = hist[-self.form_window:]
        if not hist:
            return {"avg_scored": 1.4, "avg_conceded": 1.4, "ppg": 1.3}
        n = len(hist)
        return {
            "avg_scored":   sum(self._goals_for(m, team) for m in hist) / n,
            "avg_conceded": sum(self._goals_against(m, team) for m in hist) / n,
            "ppg":          sum(self._team_points(m, team) for m in hist) / n,
        }

    def _h2h(self, home_team, away_team):
        key = tuple(sorted([home_team, away_team]))
        hist = self.h2h_history[key][-self.h2h_window:]
        if not hist:
            return {"hw_rate": 0.45, "d_rate": 0.25, "aw_rate": 0.30, "avg_goals": 2.6}
        n = len(hist)
        hw = sum(1 for m in hist if
                 (m["home_team"] == home_team and m["result"] == "H") or
                 (m["away_team"] == home_team and m["result"] == "A"))
        d = sum(1 for m in hist if m["result"] == "D")
        aw = n - hw - d
        goals = sum(m["home_score"] + m["away_score"] for m in hist)
        return {"hw_rate": hw/n, "d_rate": d/n, "aw_rate": aw/n, "avg_goals": goals/n}

    def _rest_days(self, team, current_date):
        last = self.last_match_date.get(team)
        if last is None:
            return 7.0
        return (current_date - last).days

    def _implied(self, odds):
        if pd.isna(odds) or odds <= 1:
            return np.nan
        return 1.0 / odds

    def build(self):
        rows = []
        for _, m in self.df.iterrows():
            home, away, date = m["home_team"], m["away_team"], m["date"]

            home_form  = self._team_form(home)
            away_form  = self._team_form(away)
            home_venue = self._venue_form(home, "H")
            away_venue = self._venue_form(away, "A")
            h2h        = self._h2h(home, away)

            row = {
                "date": date,
                "season": m["season_label"],
                "home_team": home,
                "away_team": away,

                "home_avg_scored":     home_form["avg_scored"],
                "home_avg_conceded":   home_form["avg_conceded"],
                "home_ppg":            home_form["ppg"],
                "home_avg_sot":        home_form["avg_sot"],

                "away_avg_scored":     away_form["avg_scored"],
                "away_avg_conceded":   away_form["avg_conceded"],
                "away_ppg":            away_form["ppg"],
                "away_avg_sot":        away_form["avg_sot"],

                "home_home_avg_scored":   home_venue["avg_scored"],
                "home_home_avg_conceded": home_venue["avg_conceded"],
                "away_away_avg_scored":   away_venue["avg_scored"],
                "away_away_avg_conceded": away_venue["avg_conceded"],

                "h2h_hw_rate":  h2h["hw_rate"],
                "h2h_d_rate":   h2h["d_rate"],
                "h2h_aw_rate":  h2h["aw_rate"],
                "h2h_avg_goals": h2h["avg_goals"],

                "home_rest_days": self._rest_days(home, date),
                "away_rest_days": self._rest_days(away, date),

                # 1X2 odds + implied
                "odds_home": m.get("odds_home"),
                "odds_draw": m.get("odds_draw"),
                "odds_away": m.get("odds_away"),
                "impl_home": self._implied(m.get("odds_home")),
                "impl_draw": self._implied(m.get("odds_draw")),
                "impl_away": self._implied(m.get("odds_away")),

                # ★ NEW: Over/Under odds + implied
                "odds_over_2_5":  m.get("odds_over_2_5"),
                "odds_under_2_5": m.get("odds_under_2_5"),
                "impl_over_2_5":  self._implied(m.get("odds_over_2_5")),
                "impl_under_2_5": self._implied(m.get("odds_under_2_5")),

                # Targets
                "target_result":  m["result"],
                "target_home_score": int(m["home_score"]),
                "target_away_score": int(m["away_score"]),
                "target_over_2_5":   int(m["over_2_5"]),
                "target_btts":       int(m["btts"]),
            }
            rows.append(row)

            # Update history AFTER features (no leakage)
            self.team_history[home].append(m)
            self.team_history[away].append(m)
            self.h2h_history[tuple(sorted([home, away]))].append(m)
            self.last_match_date[home] = date
            self.last_match_date[away] = date

        return pd.DataFrame(rows)


if __name__ == "__main__":
    import os
    os.makedirs("data/processed", exist_ok=True)
    print("Loading clean data...")
    df = pd.read_csv("data/processed/pl_clean.csv", parse_dates=["date"])
    print(f"  → {len(df)} matches")

    print("Engineering features...")
    fe = FeatureEngineer(df, form_window=5, h2h_window=5)
    features = fe.build()

    out = "data/processed/features.csv"
    features.to_csv(out, index=False)
    print(f"\n✅ Features saved → {out}")
    print(f"    Shape: {features.shape}")

    print("\n--- Over/Under odds coverage ---")
    print(f"  odds_over_2_5  non-null: {features['odds_over_2_5'].notna().sum()}/{len(features)}")
    print(f"  odds_under_2_5 non-null: {features['odds_under_2_5'].notna().sum()}/{len(features)}")
