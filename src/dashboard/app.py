import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------- Page config ----------
st.set_page_config(
    page_title="EPL Betting Analyzer",
    page_icon="⚽",
    layout="wide",
)

# ---------- Constants ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

FEATURE_COLS = [
    "home_avg_scored", "home_avg_conceded", "home_ppg", "home_avg_sot",
    "away_avg_scored", "away_avg_conceded", "away_ppg", "away_avg_sot",
    "home_home_avg_scored", "home_home_avg_conceded",
    "away_away_avg_scored", "away_away_avg_conceded",
    "h2h_hw_rate", "h2h_d_rate", "h2h_aw_rate", "h2h_avg_goals",
    "home_rest_days", "away_rest_days",
    "impl_home", "impl_draw", "impl_away",
]

RESULT_UNMAP = {0: "A", 1: "D", 2: "H"}
RESULT_LABEL = {"H": "Home Win", "D": "Draw", "A": "Away Win"}


# ---------- Loaders ----------
@st.cache_resource
def load_models():
    outcome = joblib.load(os.path.join(PROJECT_ROOT, "models/outcome_model.pkl"))
    over    = joblib.load(os.path.join(PROJECT_ROOT, "models/over_2_5_model.pkl"))
    btts    = joblib.load(os.path.join(PROJECT_ROOT, "models/btts_model.pkl"))
    return outcome, over, btts


@st.cache_data
def load_features():
    return pd.read_csv(
        os.path.join(PROJECT_ROOT, "data/processed/features.csv"),
        parse_dates=["date"],
    )


@st.cache_data
def load_backtest(name):
    path = os.path.join(PROJECT_ROOT, f"results/backtest_{name}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["date"])


# ---------- Helpers ----------
def latest_team_stats(features, team):
    """Return the most recent feature row where this team played."""
    mask = (features["home_team"] == team) | (features["away_team"] == team)
    sub = features[mask].sort_values("date")
    if sub.empty:
        return None
    last = sub.iloc[-1]
    is_home = last["home_team"] == team

    return {
        "avg_scored":   last["home_avg_scored"]   if is_home else last["away_avg_scored"],
        "avg_conceded": last["home_avg_conceded"] if is_home else last["away_avg_conceded"],
        "ppg":          last["home_ppg"]           if is_home else last["away_ppg"],
        "avg_sot":      last["home_avg_sot"]       if is_home else last["away_avg_sot"],
        "venue_scored":   last["home_home_avg_scored"]   if is_home else last["away_away_avg_scored"],
        "venue_conceded": last["home_home_avg_conceded"] if is_home else last["away_away_avg_conceded"],
    }


def h2h_stats(features, home, away, window=5):
    mask = (
        ((features["home_team"] == home) & (features["away_team"] == away)) |
        ((features["home_team"] == away) & (features["away_team"] == home))
    )
    sub = features[mask].sort_values("date").tail(window)
    if sub.empty:
        return {"hw_rate": 0.45, "d_rate": 0.25, "aw_rate": 0.30, "avg_goals": 2.6}
    hw = ((sub["home_team"] == home) & (sub["target_result"] == "H")).sum() + \
         ((sub["away_team"] == home) & (sub["target_result"] == "A")).sum()
    d = (sub["target_result"] == "D").sum()
    aw = len(sub) - hw - d
    avg_goals = (sub["target_home_score"] + sub["target_away_score"]).mean()
    return {
        "hw_rate": hw / len(sub),
        "d_rate": d / len(sub),
        "aw_rate": aw / len(sub),
        "avg_goals": avg_goals,
    }


def build_feature_row(home_stats, away_stats, h2h, home_rest, away_rest,
                      odds_home, odds_draw, odds_away):
    return pd.DataFrame([{
        "home_avg_scored":   home_stats["avg_scored"],
        "home_avg_conceded": home_stats["avg_conceded"],
        "home_ppg":          home_stats["ppg"],
        "home_avg_sot":      home_stats["avg_sot"],
        "away_avg_scored":   away_stats["avg_scored"],
        "away_avg_conceded": away_stats["avg_conceded"],
        "away_ppg":          away_stats["ppg"],
        "away_avg_sot":      away_stats["avg_sot"],
        "home_home_avg_scored":   home_stats["venue_scored"],
        "home_home_avg_conceded": home_stats["venue_conceded"],
        "away_away_avg_scored":   away_stats["venue_scored"],
        "away_away_avg_conceded": away_stats["venue_conceded"],
        "h2h_hw_rate":  h2h["hw_rate"],
        "h2h_d_rate":   h2h["d_rate"],
        "h2h_aw_rate":  h2h["aw_rate"],
        "h2h_avg_goals": h2h["avg_goals"],
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "impl_home": 1 / odds_home if odds_home > 1 else np.nan,
        "impl_draw": 1 / odds_draw if odds_draw > 1 else np.nan,
        "impl_away": 1 / odds_away if odds_away > 1 else np.nan,
    }])[FEATURE_COLS]


# ---------- Header ----------
st.title("⚽ EPL Betting Analyzer")
st.caption("Research tool — see disclaimer at bottom. Not financial advice.")

# ---------- Sidebar — Match input ----------
st.sidebar.header("Match Input")

features = load_features()
teams = sorted(set(features["home_team"]) | set(features["away_team"]))

home_team = st.sidebar.selectbox("Home Team", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
away_team = st.sidebar.selectbox("Away Team", teams, index=teams.index("Chelsea") if "Chelsea" in teams else 1)

if home_team == away_team:
    st.sidebar.error("Pick two different teams.")
    st.stop()

st.sidebar.subheader("Bookmaker Odds (Decimal)")
col_a, col_b, col_c = st.sidebar.columns(3)
odds_home = col_a.number_input("Home", min_value=1.01, value=2.10, step=0.05)
odds_draw = col_b.number_input("Draw", min_value=1.01, value=3.40, step=0.05)
odds_away = col_c.number_input("Away", min_value=1.01, value=3.80, step=0.05)

st.sidebar.subheader("Rest Days")
home_rest = st.sidebar.slider("Home team rest days", 0, 14, 4)
away_rest = st.sidebar.slider("Away team rest days", 0, 14, 4)

# ---------- Get team stats ----------
home_stats = latest_team_stats(features, home_team)
away_stats = latest_team_stats(features, away_team)
h2h = h2h_stats(features, home_team, away_team)

if home_stats is None or away_stats is None:
    st.error(f"No historical data for {home_team if home_stats is None else away_team}.")
    st.stop()

# ---------- Model predictions ----------
outcome_model, over_model, btts_model = load_models()

row = build_feature_row(home_stats, away_stats, h2h, home_rest, away_rest,
                        odds_home, odds_draw, odds_away)

# Outcome
outcome_probs = outcome_model.predict_proba(row)[0]
outcome_classes = [int(c) for c in outcome_model.classes_]
outcome_probs_dict = {RESULT_UNMAP[c]: p for c, p in zip(outcome_classes, outcome_probs)}

# O/U 2.5
p_over = float(over_model.predict_proba(row)[0][1])
p_under = 1 - p_over

# BTTS
p_btts = float(btts_model.predict_proba(row)[0][1])

# ---------- Main layout ----------
left, right = st.columns([1, 1])

with left:
    st.subheader("📊 Model Predictions")

    # Outcome probabilities
    outcome_df = pd.DataFrame([
        {"Outcome": RESULT_LABEL[k], "Probability": f"{outcome_probs_dict[k]:.1%}"}
        for k in ["H", "D", "A"]
    ])
    st.dataframe(outcome_df, hide_index=True, use_container_width=True)

    top_outcome = max(outcome_probs_dict, key=outcome_probs_dict.get)
    st.metric("Most Likely Outcome",
              RESULT_LABEL[top_outcome],
              f"{outcome_probs_dict[top_outcome]:.1%}")

    # Goal markets
    col1, col2 = st.columns(2)
    col1.metric("Over 2.5 Goals", f"{p_over:.1%}")
    col2.metric("Both Teams to Score", f"{p_btts:.1%}")

with right:
    st.subheader("💰 Value Analysis")

    implied = {"H": 1/odds_home, "D": 1/odds_draw, "A": 1/odds_away}
    odds_map = {"H": odds_home, "D": odds_draw, "A": odds_away}

    value_rows = []
    for k in ["H", "D", "A"]:
        p_model = outcome_probs_dict[k]
        p_impl = implied[k]
        abs_diff = p_model - p_impl
        value_rows.append({
            "Outcome": RESULT_LABEL[k],
            "Model": f"{p_model:.1%}",
            "Bookie Implied": f"{p_impl:.1%}",
            "Diff": f"{abs_diff:+.1%}",
            "Odds": f"{odds_map[k]:.2f}",
            "Edge?": "✅" if abs_diff >= 0.05 else "—",
        })

    st.dataframe(pd.DataFrame(value_rows), hide_index=True,
                 use_container_width=True)

    # Over/Under value
    st.markdown("**Over/Under 2.5 Goals**")
    if p_over > 0.55:
        st.info(f"Model leans **Over 2.5** ({p_over:.1%})")
    elif p_over < 0.45:
        st.info(f"Model leans **Under 2.5** ({p_under:.1%})")
    else:
        st.info("Model is neutral on Over/Under.")

# ---------- Historical backtest ----------
st.markdown("---")
st.subheader("📈 Historical Backtest Performance")

backtest_choice = st.selectbox(
    "Choose a backtest to display",
    ["no_away", "odds_3_4", "conservative", "moderate", "strict", "loose",
     "ou_tight", "ou_moderate", "ou_loose"],
    index=1,  # odds_3_4 is the profitable one
)

bt = load_backtest(backtest_choice)
if bt is None or bt.empty:
    st.warning(f"No backtest CSV found for '{backtest_choice}'.")
else:
    col1, col2, col3, col4 = st.columns(4)
    n_bets = len(bt)
    wins = (bt["result"] == "WON").sum()
    staked = bt["stake"].sum()
    net = bt["net"].sum()
    roi = net / staked if staked else 0

    col1.metric("Bets Placed", f"{n_bets}")
    col2.metric("Win Rate", f"{wins / n_bets:.1%}" if n_bets else "—")
    col3.metric("Net P/L (after tax)", f"KSh {net:,.0f}")
    col4.metric("ROI", f"{roi:+.1%}")

    # Bankroll curve
    bt_sorted = bt.sort_values("date").reset_index(drop=True)
    bt_sorted["bet_num"] = range(1, len(bt_sorted) + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bt_sorted["bet_num"],
        y=bt_sorted["bankroll_after"],
        mode="lines+markers",
        name="Bankroll",
        line=dict(color="#22c55e" if net > 0 else "#ef4444", width=2),
    ))
    fig.add_hline(y=10000, line_dash="dash", line_color="gray",
                  annotation_text="Starting bankroll")
    fig.update_layout(
        title=f"Bankroll over time — {backtest_choice}",
        xaxis_title="Bet number",
        yaxis_title="Bankroll (KSh)",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show bet log"):
        st.dataframe(bt_sorted, use_container_width=True)

# ---------- Footer disclaimer ----------
st.markdown("---")
st.warning(
    "**⚠️ Disclaimer**  \n"
    "This tool is for **research and education only**.  \n"
    "- The models were backtested on 2023–2025 Premier League data and "
    "most configurations show **negative ROI after Kenyan withholding tax**.  \n"
    "- The only positive configuration (H/D, odds 3–4) is based on a very "
    "small sample (~42 bets) and may not generalize.  \n"
    "- No model on this data has demonstrated a reliable edge against the "
    "bookmaker margin.  \n"
    "- **Do not bet real money on the basis of this tool alone.**"
)
