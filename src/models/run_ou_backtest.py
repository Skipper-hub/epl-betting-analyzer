import os
import joblib
import numpy as np
import pandas as pd

from src.models.value_betting import KENYAN_TAX
from src.models.train import load_features, split_chronological


class OUValueFinder:
    """
    Value finder for Over/Under 2.5 markets.
    Uses absolute diff, not relative edge.
    """

    def __init__(self, min_abs_diff=0.05, min_model_prob=0.40,
                 max_model_prob=0.70, min_odds=1.60, max_odds=2.80,
                 kelly_fraction=0.25, max_stake_pct=0.03):
        self.min_abs_diff = min_abs_diff
        self.min_model_prob = min_model_prob
        self.max_model_prob = max_model_prob
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.kelly_fraction = kelly_fraction
        self.max_stake_pct = max_stake_pct

    @staticmethod
    def implied_prob(odds):
        if pd.isna(odds) or odds <= 1:
            return np.nan
        return 1.0 / odds

    def kelly_stake(self, p, odds):
        b = odds - 1
        if b <= 0:
            return 0.0
        f = (p * b - (1 - p)) / b
        return max(0.0, f)

    def find_bet(self, p_over, odds_over, odds_under):
        candidates = []

        # OVER
        if not pd.isna(odds_over) and self.min_odds <= odds_over <= self.max_odds:
            implied = self.implied_prob(odds_over)
            diff = p_over - implied
            if (self.min_model_prob <= p_over <= self.max_model_prob
                    and diff >= self.min_abs_diff):
                stake = min(self.kelly_stake(p_over, odds_over)
                            * self.kelly_fraction, self.max_stake_pct)
                if stake > 0:
                    candidates.append({
                        "outcome": "OVER", "odds": odds_over,
                        "model_prob": p_over, "implied_prob": implied,
                        "abs_diff": diff, "stake_pct": stake,
                    })

        # UNDER
        p_under = 1 - p_over
        if not pd.isna(odds_under) and self.min_odds <= odds_under <= self.max_odds:
            implied = self.implied_prob(odds_under)
            diff = p_under - implied
            if (self.min_model_prob <= p_under <= self.max_model_prob
                    and diff >= self.min_abs_diff):
                stake = min(self.kelly_stake(p_under, odds_under)
                            * self.kelly_fraction, self.max_stake_pct)
                if stake > 0:
                    candidates.append({
                        "outcome": "UNDER", "odds": odds_under,
                        "model_prob": p_under, "implied_prob": implied,
                        "abs_diff": diff, "stake_pct": stake,
                    })

        return max(candidates, key=lambda c: c["abs_diff"]) if candidates else None


def run_ou_backtest(model, feature_cols, test, finder, bankroll=10000):
    test = test.sort_values("date").reset_index(drop=True)
    bets = []
    bank = bankroll

    for _, row in test.iterrows():
        X = row[feature_cols].to_frame().T.astype(float)
        p_over = model.predict_proba(X)[0][1]

        bet = finder.find_bet(
            p_over,
            row.get("odds_over_2_5"),
            row.get("odds_under_2_5"),
        )
        if bet is None:
            continue

        stake = bank * bet["stake_pct"]
        if stake < 1:
            continue

        actual_over = int(row["target_over_2_5"])
        bet_is_over = (bet["outcome"] == "OVER")
        won = (bet_is_over and actual_over == 1) or (not bet_is_over and actual_over == 0)

        net = stake * (bet["odds"] - 1) * (1 - KENYAN_TAX) if won else -stake
        bank += net

        bets.append({
            "date": row["date"], "season": row["season"],
            "home_team": row["home_team"], "away_team": row["away_team"],
            "bet_outcome": bet["outcome"], "odds": bet["odds"],
            "model_prob": bet["model_prob"], "implied_prob": bet["implied_prob"],
            "abs_diff": bet["abs_diff"], "stake": round(stake, 2),
            "actual_result": "OVER" if actual_over else "UNDER",
            "result": "WON" if won else "LOST",
            "net": round(net, 2), "bankroll_after": round(bank, 2),
        })

    return pd.DataFrame(bets), bank


def summarise(bets_df, initial_bankroll):
    if bets_df.empty:
        print("\n⚠️  No bets placed. Filters too strict.")
        return
    n = len(bets_df)
    wins = (bets_df["result"] == "WON").sum()
    staked = bets_df["stake"].sum()
    net = bets_df["net"].sum()
    roi = net / staked if staked else 0
    final = initial_bankroll + net

    print("\n" + "=" * 60)
    print("OU BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Bets placed:            {n}")
    print(f"Wins / Losses:          {wins} / {n - wins}")
    print(f"Win rate:               {wins/n:.1%}")
    print(f"Average abs_diff:       {bets_df['abs_diff'].mean():+.3f}")
    print(f"Average odds:           {bets_df['odds'].mean():.2f}")
    print(f"Total staked:           KSh {staked:>12,.2f}")
    print(f"Net profit (after tax): KSh {net:>12,.2f}")
    print(f"ROI:                    {roi:+.2%}")
    print(f"Final bankroll:         KSh {final:>12,.2f}")

    print("\n--- By side ---")
    for side in ["OVER", "UNDER"]:
        sub = bets_df[bets_df["bet_outcome"] == side]
        if sub.empty:
            print(f"  {side}: no bets")
            continue
        sub_roi = sub["net"].sum() / sub["stake"].sum()
        print(f"  {side}: {len(sub):>4} bets | "
              f"win rate {(sub['result']=='WON').mean():.1%} | "
              f"ROI {sub_roi:+.1%}")

    print("\n--- By season ---")
    for season in sorted(bets_df["season"].unique()):
        sub = bets_df[bets_df["season"] == season]
        sub_roi = sub["net"].sum() / sub["stake"].sum()
        print(f"  {season}: {len(sub):>4} bets | ROI {sub_roi:+.1%} | "
              f"Net KSh {sub['net'].sum():>10,.2f}")

    print("\n--- By odds bucket ---")
    b = bets_df.copy()
    b["odds_bucket"] = pd.cut(b["odds"], [1.5, 1.8, 2.0, 2.2, 2.5, 3.0])
    for bucket, sub in b.groupby("odds_bucket", observed=True):
        if sub.empty:
            continue
        sub_roi = sub["net"].sum() / sub["stake"].sum()
        print(f"  {bucket}: {len(sub):>4} bets | "
              f"win rate {(sub['result']=='WON').mean():.1%} | "
              f"ROI {sub_roi:+.1%}")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    df = load_features()
    _, test = split_chronological(df)

    if "odds_over_2_5" not in test.columns:
        raise SystemExit(
            "\n❌ features.csv is missing odds_over_2_5 — re-run the cleaner "
            "and feature engineering first."
        )

    model = joblib.load("models/over_2_5_model.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")

    experiments = [
        ("ou_tight", OUValueFinder(
            min_abs_diff=0.07, min_model_prob=0.45, max_model_prob=0.65,
            min_odds=1.60, max_odds=2.40,
        )),
        ("ou_moderate", OUValueFinder(
            min_abs_diff=0.05, min_model_prob=0.40, max_model_prob=0.70,
            min_odds=1.60, max_odds=2.80,
        )),
        ("ou_loose", OUValueFinder(
            min_abs_diff=0.03, min_model_prob=0.35, max_model_prob=0.75,
            min_odds=1.50, max_odds=3.00,
        )),
    ]

    for label, finder in experiments:
        print("\n" + "#" * 60)
        print(f"# OU BACKTEST: {label.upper()}")
        print(f"#   abs_diff>={finder.min_abs_diff}  odds=[{finder.min_odds},{finder.max_odds}]  "
              f"prob=[{finder.min_model_prob},{finder.max_model_prob}]")
        print("#" * 60)

        bets, _ = run_ou_backtest(model, feature_cols, test, finder)
        bets.to_csv(f"results/backtest_{label}.csv", index=False)
        summarise(bets, 10000)
        print(f"→ results/backtest_{label}.csv")
