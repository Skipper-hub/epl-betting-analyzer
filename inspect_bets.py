import pandas as pd

df = pd.read_csv("results/backtest_strict.csv")

print(f"Total bets: {len(df)}")

print("\n=== Odds distribution ===")
print(df["odds"].describe().round(3))

print("\n=== Win rate by odds bucket ===")
df["odds_bucket"] = pd.cut(df["odds"], [1, 2, 3, 5, 10, 100])
print(df.groupby("odds_bucket", observed=True).agg(
    n=("odds", "size"),
    wins=("result", lambda s: (s == "WON").sum()),
    win_rate=("result", lambda s: (s == "WON").mean()),
    avg_stake=("stake", "mean"),
    net=("net", "sum"),
).round(3))

print("\n=== Model probability distribution on bets ===")
print(df["model_prob"].describe().round(3))

print("\n=== Implied probability distribution on bets ===")
print(df["implied_prob"].describe().round(3))

print("\n=== Top 10 highest-edge LOSING bets ===")
losers = df[df["result"] == "LOST"].nlargest(10, "edge")
cols = ["date", "home_team", "away_team", "bet_outcome", "odds",
        "model_prob", "implied_prob", "edge", "actual_result", "net"]
print(losers[cols].to_string(index=False))

print("\n=== Top 10 highest-edge WINNING bets ===")
winners = df[df["result"] == "WON"].nlargest(10, "edge")
if len(winners) == 0:
    print("(no winning bets)")
else:
    print(winners[cols].to_string(index=False))

print("\n=== Summary ===")
print(f"Bets where implied_prob < 0.15 (heavy longshots): "
      f"{(df['implied_prob'] < 0.15).sum()} "
      f"({(df['implied_prob'] < 0.15).mean():.1%} of all bets)")
print(f"Bets where model_prob < 0.20: "
      f"{(df['model_prob'] < 0.20).sum()} "
      f"({(df['model_prob'] < 0.20).mean():.1%} of all bets)")
print(f"Median odds on bets: {df['odds'].median():.2f}")
