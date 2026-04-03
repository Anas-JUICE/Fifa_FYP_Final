from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

import pandas as pd
import streamlit as st

from utils.helpers import load_team_profiles, predict_match_proba, load_json, BEST_MODEL_SUMMARY_PATH

st.set_page_config(page_title="FIFA World Cup Match Predictor", layout="wide")
st.title("FIFA World Cup Match Predictor")

profiles_df = load_team_profiles()
profiles_dict = {row["team"]: row for _, row in profiles_df.iterrows()}
teams = sorted(profiles_df["team"].tolist())
best_summary = load_json(BEST_MODEL_SUMMARY_PATH) if BEST_MODEL_SUMMARY_PATH.exists() else {"best_model_name": "best_model_not_generated_yet"}

st.caption(f"Selected best model: {best_summary['best_model_name']}")

col1, col2 = st.columns(2)

with col1:
    team_a = st.selectbox("Team A", teams, index=0)

with col2:
    default_index = 1 if len(teams) > 1 else 0
    team_b = st.selectbox("Team B", teams, index=default_index)

settings_col1, settings_col2, settings_col3 = st.columns(3)
with settings_col1:
    neutral = st.checkbox("Neutral ground", value=True)
with settings_col2:
    match_importance = st.slider("Match importance", min_value=1.0, max_value=5.0, value=5.0, step=1.0)
with settings_col3:
    tournament = st.text_input("Tournament", value="FIFA World Cup")

def profile_table(team_name: str):
    row = profiles_dict[team_name]
    return pd.DataFrame({
        "Metric": ["Elo", "Wins Last 5", "Draws Last 5", "Losses Last 5", "GF Avg Last 5", "GA Avg Last 5", "GD Avg Last 5", "Last Seen Date"],
        "Value": [row["elo"], row["wins_last5"], row["draws_last5"], row["loss_last5"], row["gf_avg_last5"], row["ga_avg_last5"], row["gd_avg_last5"], row["last_seen_date"]]
    })

profile_col1, profile_col2 = st.columns(2)
with profile_col1:
    st.subheader(team_a)
    st.dataframe(profile_table(team_a), use_container_width=True, hide_index=True)
with profile_col2:
    st.subheader(team_b)
    st.dataframe(profile_table(team_b), use_container_width=True, hide_index=True)

if st.button("Predict Match Outcome", use_container_width=True):
    if team_a == team_b:
        st.error("Please choose two different teams.")
    else:
        result = predict_match_proba(
            team_a=team_a,
            team_b=team_b,
            model_name="best",
            tournament=tournament,
            neutral=neutral,
            match_importance=match_importance
        )

        probs_df = pd.DataFrame({
            "Outcome": [f"{team_a} Win", "Draw", f"{team_b} Win"],
            "Probability": [result["team_a_win"], result["draw"], result["team_b_win"]]
        })

        predicted_probability = probs_df["Probability"].max()
        st.success(f"Predicted outcome: {result['predicted_outcome']} ({predicted_probability:.2%})")

        stat1, stat2, stat3 = st.columns(3)
        with stat1:
            st.metric(f"{team_a} Win", f"{result['team_a_win']:.2%}")
        with stat2:
            st.metric("Draw", f"{result['draw']:.2%}")
        with stat3:
            st.metric(f"{team_b} Win", f"{result['team_b_win']:.2%}")

        st.subheader("Probability Table")
        st.dataframe(probs_df, use_container_width=True, hide_index=True)

        st.subheader("Probability Chart")
        st.bar_chart(probs_df.set_index("Outcome"))

        a = profiles_dict[team_a]
        b = profiles_dict[team_b]
        insights = []

        elo_diff = float(a["elo"]) - float(b["elo"])
        if abs(elo_diff) >= 50:
            higher = team_a if elo_diff > 0 else team_b
            insights.append(f"{higher} has the stronger Elo rating by {abs(elo_diff):.0f} points.")

        gd_diff = float(a["gd_avg_last5"]) - float(b["gd_avg_last5"])
        if abs(gd_diff) >= 0.5:
            better_form = team_a if gd_diff > 0 else team_b
            insights.append(f"{better_form} comes in with the better recent goal-difference form.")

        ga_diff = float(a["ga_avg_last5"]) - float(b["ga_avg_last5"])
        if abs(ga_diff) >= 0.4:
            stronger_defense = team_a if ga_diff < 0 else team_b
            insights.append(f"{stronger_defense} looks stronger defensively based on recent goals conceded.")

        if not insights:
            insights.append("The matchup looks balanced, which usually increases uncertainty and draw chance.")

        st.subheader("Matchup Insights")
        for item in insights:
            st.write(f"- {item}")
