import os
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, log_loss, classification_report, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV


# ------- Feature / target columns -------
FEATURE_COLS = [
    "home_avg_scored", "home_avg_conceded", "home_ppg", "home_avg_sot",
    "away_avg_scored", "away_avg_conceded", "away_ppg", "away_avg_sot",
    "home_home_avg_scored", "home_home_avg_conceded",
    "away_away_avg_scored", "away_away_avg_conceded",
    "h2h_hw_rate", "h2h_d_rate", "h2h_aw_rate", "h2h_avg_goals",
    "home_rest_days", "away_rest_days",
    "impl_home", "impl_draw", "impl_away",
]

# XGBoost needs integer labels. We map and keep the mapping.
RESULT_MAP   = {"A": 0, "D": 1, "H": 2}
RESULT_UNMAP = {v: k for k, v in RESULT_MAP.items()}

TRAIN_SEASONS = {"2020/21", "2021/22", "2022/23"}
TEST_SEASONS  = {"2023/24", "2024/25"}

# ★ NEW: calibration method — switch from "isotonic" to "sigmoid"
CALIBRATION_METHOD = "sigmoid"


def load_features(path="data/processed/features.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["target_result_enc"] = df["target_result"].map(RESULT_MAP)
    return df


def split_chronological(df):
    train = df[df["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
    test  = df[df["season"].isin(TEST_SEASONS)].reset_index(drop=True)
    print(f"Train: {len(train)} matches  |  Test: {len(test)} matches")
    print(f"  Train seasons: {sorted(train['season'].unique())}")
    print(f"  Test seasons:  {sorted(test['season'].unique())}")
    return train, test


def make_xgb_multiclass():
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )


def make_xgb_binary():
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


# -------------- Outcome (H/D/A) --------------

def train_outcome(train, test):
    print("\n" + "=" * 60)
    print("MODEL 1: Match outcome (H/D/A)")
    print("=" * 60)

    X_train = train[FEATURE_COLS]
    y_train = train["target_result_enc"]
    X_test  = test[FEATURE_COLS]
    y_test  = test["target_result_enc"]

    base = make_xgb_multiclass()
    base.fit(X_train, y_train)

    model = CalibratedClassifierCV(base, method=CALIBRATION_METHOD, cv=3)
    model.fit(X_train, y_train)

    preds_enc = model.predict(X_test)
    probs     = model.predict_proba(X_test)

    acc = accuracy_score(y_test, preds_enc)
    ll  = log_loss(y_test, probs, labels=list(model.classes_))

    print(f"Calibration: {CALIBRATION_METHOD}")
    print(f"Accuracy: {acc:.3f}  |  Log-loss: {ll:.3f}")

    y_test_str  = [RESULT_UNMAP[int(v)] for v in y_test]
    preds_str   = [RESULT_UNMAP[int(v)] for v in preds_enc]
    classes_str = [RESULT_UNMAP[int(c)] for c in model.classes_]

    print("\nClassification report:")
    print(classification_report(y_test_str, preds_str,
                                labels=classes_str, digits=3))

    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(
        confusion_matrix(y_test_str, preds_str, labels=classes_str),
        index=[f"true_{c}" for c in classes_str],
        columns=[f"pred_{c}" for c in classes_str],
    ))

    baseline = (np.array(y_test_str) == "H").mean()
    print(f"\nBaseline (always 'H'): {baseline:.3f}  |  Model: {acc:.3f}  "
          f"({'+' if acc >= baseline else ''}{100*(acc-baseline):.1f} pp)")

    return model


# -------------- Over 2.5 --------------

def train_over_2_5(train, test):
    print("\n" + "=" * 60)
    print("MODEL 2: Over 2.5 goals")
    print("=" * 60)

    X_train = train[FEATURE_COLS]
    y_train = train["target_over_2_5"]
    X_test  = test[FEATURE_COLS]
    y_test  = test["target_over_2_5"]

    base = make_xgb_binary()
    base.fit(X_train, y_train)

    model = CalibratedClassifierCV(base, method=CALIBRATION_METHOD, cv=3)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    ll  = log_loss(y_test, probs)

    print(f"Calibration: {CALIBRATION_METHOD}")
    print(f"Accuracy: {acc:.3f}  |  Log-loss: {ll:.3f}")
    print(f"Base rate (over 2.5): {y_test.mean():.3f}")

    return model


# -------------- BTTS --------------

def train_btts(train, test):
    print("\n" + "=" * 60)
    print("MODEL 3: Both Teams To Score")
    print("=" * 60)

    X_train = train[FEATURE_COLS]
    y_train = train["target_btts"]
    X_test  = test[FEATURE_COLS]
    y_test  = test["target_btts"]

    base = make_xgb_binary()
    base.fit(X_train, y_train)

    model = CalibratedClassifierCV(base, method=CALIBRATION_METHOD, cv=3)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    ll  = log_loss(y_test, probs)

    print(f"Calibration: {CALIBRATION_METHOD}")
    print(f"Accuracy: {acc:.3f}  |  Log-loss: {ll:.3f}")
    print(f"Base rate (BTTS): {y_test.mean():.3f}")

    return model


# -------------- Feature importance --------------

def report_importance(train, top_n=10):
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (raw XGBoost, no calibration)")
    print("=" * 60)

    base = make_xgb_multiclass()
    base.fit(train[FEATURE_COLS], train["target_result_enc"])

    imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": base.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(imp.head(top_n).to_string(index=False))
    return imp


# -------------- Main --------------

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    df = load_features()
    train, test = split_chronological(df)

    outcome_model = train_outcome(train, test)
    over_model    = train_over_2_5(train, test)
    btts_model    = train_btts(train, test)

    report_importance(train)

    joblib.dump(outcome_model,     "models/outcome_model.pkl")
    joblib.dump(over_model,        "models/over_2_5_model.pkl")
    joblib.dump(btts_model,        "models/btts_model.pkl")
    joblib.dump(FEATURE_COLS,      "models/feature_cols.pkl")
    joblib.dump(RESULT_UNMAP,      "models/result_unmap.pkl")

    print("\n✅ Saved models → models/")
