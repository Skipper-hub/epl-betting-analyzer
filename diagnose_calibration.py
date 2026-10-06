import joblib
import numpy as np
import pandas as pd
from src.models.train import load_features, split_chronological, FEATURE_COLS, RESULT_UNMAP

df = load_features()
_, test = split_chronological(df)
model = joblib.load("models/outcome_model.pkl")
unmap = {int(k): v for k, v in joblib.load("models/result_unmap.pkl").items()}

X = test[FEATURE_COLS].astype(float)
probs = model.predict_proba(X)
classes = [int(c) for c in model.classes_]

# For each predicted outcome, get its assigned probability
top_idx = probs.argmax(axis=1)
top_outcome = [unmap[classes[i]] for i in top_idx]
top_prob    = probs.max(axis=1)
correct     = np.array([top_outcome[i] == test["target_result"].iloc[i]
                        for i in range(len(test))])

print("=" * 60)
print("CALIBRATION DIAGNOSTIC — top-pick probabilities")
print("=" * 60)
print(f"{'prob_bin':<15}{'n':>6}{'avg_pred':>12}{'actual':>12}{'gap':>10}")
print("-" * 60)

for lo, hi in [(0.0,0.35),(0.35,0.4),(0.4,0.45),(0.45,0.5),
               (0.5,0.55),(0.55,0.6),(0.6,0.7),(0.7,1.0)]:
    mask = (top_prob >= lo) & (top_prob < hi)
    n = mask.sum()
    if n == 0:
        continue
    avg_pred = top_prob[mask].mean()
    actual   = correct[mask].mean()
    gap = avg_pred - actual
    print(f"{lo:.2f}-{hi:.2f}      {n:>6}{avg_pred:>12.3f}{actual:>12.3f}{gap:>+10.3f}")

print("\n" + "=" * 60)
print("IMPLIED PROB vs MODEL PROB (per outcome)")
print("=" * 60)
for outcome, col_impl in [("H","impl_home"),("D","impl_draw"),("A","impl_away")]:
    idx = classes.index({"H":2,"D":1,"A":0}[outcome])
    model_p = probs[:, idx]
    implied_p = test[col_impl].values
    print(f"\nOutcome {outcome}:")
    print(f"  mean model prob:   {model_p.mean():.3f}")
    print(f"  mean implied prob: {implied_p.mean():.3f}")
    print(f"  mean gap (model - implied): {(model_p - implied_p).mean():+.3f}")

print("\n" + "=" * 60)
print("HOW OFTEN DOES MODEL DISAGREE WITH MARKET?")
print("=" * 60)
top_implied = test[["impl_home","impl_draw","impl_away"]].values.argmax(axis=1)
# Map: 0=H, 1=D, 2=A in test column order
top_implied_letter = [["H","D","A"][i] for i in top_implied]
agree = np.array([top_outcome[i] == top_implied_letter[i] for i in range(len(test))])
print(f"Model top pick == Market top pick: {agree.mean():.1%}")
print(f"Model disagrees with market on: {100*(1-agree.mean()):.1f}% of matches")
