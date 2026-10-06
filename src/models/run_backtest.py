import os
import joblib
import pandas as pd

from src.models.value_betting import (
    ValueBetFinder, Backtester, summarise_backtest
)
from src.models.train import load_features, split_chronological


def main():
    os.makedirs("results", exist_ok=True)

    print("Loading features...")
    df = load_features()
    _, test = split_chronological(df)

    print("\nLoading trained model...")
    model = joblib.load("models/outcome_model.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    result_unmap = joblib.load("models/result_unmap.pkl")

    experiments = [
        # A — only the profitable odds range
        ("odds_3_4", ValueBetFinder(
            min_abs_diff=0.05,
            min_model_prob=0.25,
            max_model_prob=0.65,
            min_odds=3.00,
            max_odds=4.00,
            kelly_fraction=0.25,
            max_stake_pct=0.03,
        )),

        # B — same as previous no_away but lower threshold for draws
        #     (we'll simulate by allowing more edge-tolerant bets overall)
        ("tight_plus_draws", ValueBetFinder(
            min_abs_diff=0.03,          # lower threshold -> more draw bets slip in
            min_model_prob=0.25,
            max_model_prob=0.55,        # draws have prob ~0.20-0.35; this helps
            min_odds=2.50,
            max_odds=5.00,
            kelly_fraction=0.25,
            max_stake_pct=0.03,
        )),

        # C — odds 3-4 + allow lower model_prob (catches draws)
        ("odds_3_4_draws", ValueBetFinder(
            min_abs_diff=0.03,
            min_model_prob=0.20,
            max_model_prob=0.60,
            min_odds=2.80,
            max_odds=4.50,
            kelly_fraction=0.25,
            max_stake_pct=0.03,
        )),
    ]

    for label, finder in experiments:
        print("\n" + "#" * 60)
        print(f"# BACKTEST: {label.upper()}")
        print(f"#   abs_diff>={finder.min_abs_diff}  "
              f"odds=[{finder.min_odds}, {finder.max_odds}]  "
              f"prob=[{finder.min_model_prob}, {finder.max_model_prob}]")
        print("#" * 60)

        bt = Backtester(
            outcome_model=model,
            feature_cols=feature_cols,
            result_unmap=result_unmap,
            bankroll=10000,
            finder=finder,
            verbose=False,          # quieter, since we have 3 runs
        )
        bets = bt.run(test)

        out_csv = f"results/backtest_{label}.csv"
        bets.to_csv(out_csv, index=False)

        summarise_backtest(bets, initial_bankroll=10000)
        print(f"→ {out_csv}")


if __name__ == "__main__":
    main()
