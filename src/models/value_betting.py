import os
import joblib
import numpy as np
import pandas as pd


# ----- Constants -----
KENYAN_TAX = 0.20
RESULT_MAP = {"A": 0, "D": 1, "H": 2}

# ★ NEW: outcomes we allow betting on. Home + Draw only — away bets lose money.
ALLOWED_OUTCOMES = ["H", "D"]


class ValueBetFinder:
    """
    Given a model probability for an outcome and bookie decimal odds,
    decide whether to bet and how much.

    Design (learned from backtests):
      - Home + Draw only. Away bets consistently lose ~-40% ROI.
      - Use ABSOLUTE probability difference, not relative edge.
      - Cap odds range to avoid longshots.
      - Require a minimum model probability.
    """

    def __init__(
        self,
        min_abs_diff=0.05,
        min_model_prob=0.25,
        max_model_prob=0.65,
        min_odds=2.00,
        max_odds=5.00,
        kelly_fraction=0.25,
        max_stake_pct=0.03,
    ):
        self.min_abs_diff   = min_abs_diff
        self.min_model_prob = min_model_prob
        self.max_model_prob = max_model_prob
        self.min_odds       = min_odds
        self.max_odds       = max_odds
        self.kelly_fraction = kelly_fraction
        self.max_stake_pct  = max_stake_pct

    @staticmethod
    def implied_prob(decimal_odds):
        if pd.isna(decimal_odds) or decimal_odds <= 1:
            return np.nan
        return 1.0 / decimal_odds

    def kelly_stake(self, model_prob, decimal_odds):
        b = decimal_odds - 1.0
        p = model_prob
        q = 1 - p
        if b <= 0:
            return 0.0
        f = (p * b - q) / b
        return max(0.0, f)

    def find_bet(self, model_probs, odds_by_outcome):
        candidates = []
        for outcome in ALLOWED_OUTCOMES:            # ★ only H and D
            p   = model_probs.get(outcome)
            odd = odds_by_outcome.get(outcome)
            if p is None or odd is None or pd.isna(odd):
                continue

            # ---- filters ----
            if not (self.min_odds <= odd <= self.max_odds):
                continue
            if not (self.min_model_prob <= p <= self.max_model_prob):
                continue

            implied = self.implied_prob(odd)
            abs_diff = p - implied
            if abs_diff < self.min_abs_diff:
                continue

            k = self.kelly_stake(p, odd)
            stake_pct = min(k * self.kelly_fraction, self.max_stake_pct)
            if stake_pct <= 0:
                continue

            candidates.append({
                "outcome": outcome,
                "model_prob": p,
                "odds": odd,
                "implied_prob": implied,
                "abs_diff": abs_diff,
                "stake_pct": stake_pct,
            })

        if not candidates:
            return None
        return max(candidates, key=lambda c: c["abs_diff"])


class Backtester:
    def __init__(self, outcome_model, feature_cols, result_unmap,
                 bankroll=10000, finder=None, verbose=True):
        self.model = outcome_model
        self.feature_cols = feature_cols
        self.result_unmap = {int(k): v for k, v in result_unmap.items()}
        self.bankroll = bankroll
        self.initial_bankroll = bankroll
        self.finder = finder or ValueBetFinder()
        self.verbose = verbose
        self.bets = []

    def _predict_probs(self, row):
        X = row[self.feature_cols].to_frame().T.astype(float)
        probs = self.model.predict_proba(X)[0]
        classes = [int(c) for c in self.model.classes_]
        return {self.result_unmap[c]: p for c, p in zip(classes, probs)}

    def run(self, test_df):
        test_df = test_df.sort_values("date").reset_index(drop=True)

        for i, row in test_df.iterrows():
            model_probs = self._predict_probs(row)
            odds_by_outcome = {
                "H": row["odds_home"],
                "D": row["odds_draw"],
                "A": row["odds_away"],
            }

            bet = self.finder.find_bet(model_probs, odds_by_outcome)
            if bet is None:
                continue

            stake = self.bankroll * bet["stake_pct"]
            if stake < 1:
                continue

            actual = row["target_result"]
            won = (bet["outcome"] == actual)

            if won:
                gross = stake * (bet["odds"] - 1)
                net   = gross * (1 - KENYAN_TAX)
                self.bankroll += net
                result = "WON"
            else:
                net = -stake
                self.bankroll += net
                result = "LOST"

            self.bets.append({
                "date": row["date"],
                "season": row["season"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "bet_outcome": bet["outcome"],
                "odds": bet["odds"],
                "model_prob": bet["model_prob"],
                "implied_prob": bet["implied_prob"],
                "abs_diff": bet["abs_diff"],
                "stake": round(stake, 2),
                "actual_result": actual,
                "result": result,
                "net": round(net, 2),
                "bankroll_after": round(self.bankroll, 2),
            })

            if self.verbose and (i % 100 == 0):
                print(f"  Match {i:>4} | Bankroll: {self.bankroll:>10,.2f} "
                      f"| Bets: {len(self.bets)}")

        return pd.DataFrame(self.bets)


def summarise_backtest(bets_df, initial_bankroll):
    if bets_df.empty:
        print("\n⚠️  No bets were placed. Filters are too strict.")
        return {}

    n_bets   = len(bets_df)
    n_wins   = (bets_df["result"] == "WON").sum()
    win_rate = n_wins / n_bets
    total_staked = bets_df["stake"].sum()
    net_profit   = bets_df["net"].sum()
    roi          = net_profit / total_staked
    final_bank   = initial_bankroll + net_profit
    avg_diff     = bets_df["abs_diff"].mean()
    avg_odds     = bets_df["odds"].mean()

    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Bets placed:            {n_bets}")
    print(f"Wins / Losses:          {n_wins} / {n_bets - n_wins}")
    print(f"Win rate:               {win_rate:.1%}")
    print(f"Average abs_diff:       {avg_diff:+.3f}")
    print(f"Average odds:           {avg_odds:.2f}")
    print(f"Total staked:           KSh {total_staked:>12,.2f}")
    print(f"Net profit (after tax): KSh {net_profit:>12,.2f}")
    print(f"ROI:                    {roi:+.2%}")
    print(f"Final bankroll:         KSh {final_bank:>12,.2f}")
    print(f"Growth:                 {(final_bank/initial_bankroll - 1):+.2%}")

    print("\n--- By outcome bet on ---")
    for outcome in ["H", "D", "A"]:
        sub = bets_df[bets_df["bet_outcome"] == outcome]
        if sub.empty:
            print(f"  {outcome}: no bets")
            continue
        sub_roi = sub["net"].sum() / sub["stake"].sum()
        print(f"  {outcome}: {len(sub):>4} bets | "
              f"win rate {(sub['result']=='WON').mean():.1%} | "
              f"ROI {sub_roi:+.1%}")

    print("\n--- By season ---")
    for season in sorted(bets_df["season"].unique()):
        sub = bets_df[bets_df["season"] == season]
        sub_roi = sub["net"].sum() / sub["stake"].sum()
        print(f"  {season}: {len(sub):>4} bets | ROI {sub_roi:+.1%} | "
              f"Net KSh {sub['net'].sum():>10,.2f}")

    print("\n--- By odds bucket ---")
    bets_df = bets_df.copy()
    bets_df["odds_bucket"] = pd.cut(bets_df["odds"], [2, 3, 4, 5, 100])
    for bucket, sub in bets_df.groupby("odds_bucket", observed=True):
        if sub.empty:
            continue
        sub_roi = sub["net"].sum() / sub["stake"].sum()
        print(f"  {bucket}: {len(sub):>4} bets | "
              f"win rate {(sub['result']=='WON').mean():.1%} | "
              f"ROI {sub_roi:+.1%}")

    return {
        "n_bets": n_bets, "win_rate": win_rate, "roi": roi,
        "net_profit": net_profit, "final_bankroll": final_bank,
    }
