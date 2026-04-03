import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

import streamlit as st
import pandas as pd
from utils.helpers import load_team_profiles, predict_match_proba

st.markdown("[Open deployed app link](https://fifafypfinal-sgbn94axknrx2vgiss2qlc.streamlit.app/)")
st.set_page_config(page_title="FIFA World Cup Predictor", layout="wide")
st.title("FIFA World Cup Match Predictor & Tournament Simulation")

# ---------- Load data ----------
profiles_df = load_team_profiles()
teams = profiles_df["team"].sort_values().tolist()

simulation_path = ROOT / "results" / "worldcup_simulation_results.csv"
groups_path = ROOT / "results" / "auto_seeded_groups.csv"

# ---------- Navigation ----------
page = st.sidebar.radio(
    "Go to",
    ["Match Prediction", "Tournament Simulation"]
)

# =========================================================
# MATCH PREDICTION PAGE
# =========================================================
if page == "Match Prediction":
    st.header("Match Prediction")

    col1, col2 = st.columns(2)

    with col1:
        team_a = st.selectbox("Team A", teams, index=0)

    with col2:
        default_index = 1 if len(teams) > 1 else 0
        team_b = st.selectbox("Team B", teams, index=default_index)

    neutral = st.checkbox("Neutral ground", value=True)
    match_importance = st.slider("Match importance", min_value=1.0, max_value=5.0, value=5.0, step=1.0)
    tournament = st.text_input("Tournament", value="FIFA World Cup")

    if st.button("Predict"):
        if team_a == team_b:
            st.error("Pick two different teams.")
        else:
            result = predict_match_proba(
                team_a=team_a,
                team_b=team_b,
                tournament=tournament,
                neutral=neutral,
                match_importance=match_importance
            )

            probs_df = pd.DataFrame({
                "Outcome": [f"{team_a} Win", "Draw", f"{team_b} Win"],
                "Probability": [result["team_a_win"], result["draw"], result["team_b_win"]]
            })

            winner_idx = probs_df["Probability"].idxmax()
            predicted_outcome = probs_df.loc[winner_idx, "Outcome"]
            confidence = probs_df.loc[winner_idx, "Probability"]

            st.subheader(f"{team_a} vs {team_b}")
            st.success(f"Predicted outcome: {predicted_outcome} ({confidence:.2%})")

            profile_a = profiles_df[profiles_df["team"] == team_a].iloc[0]
            profile_b = profiles_df[profiles_df["team"] == team_b].iloc[0]

            st.markdown("### Team Comparison")
            compare_df = pd.DataFrame({
                "Feature": ["Elo", "Wins (last 5)", "Draws (last 5)", "Losses (last 5)", "GF Avg (last 5)", "GA Avg (last 5)", "GD Avg (last 5)"],
                team_a: [
                    profile_a["elo"], profile_a["wins_last5"], profile_a["draws_last5"], profile_a["loss_last5"],
                    profile_a["gf_avg_last5"], profile_a["ga_avg_last5"], profile_a["gd_avg_last5"]
                ],
                team_b: [
                    profile_b["elo"], profile_b["wins_last5"], profile_b["draws_last5"], profile_b["loss_last5"],
                    profile_b["gf_avg_last5"], profile_b["ga_avg_last5"], profile_b["gd_avg_last5"]
                ]
            })
            st.dataframe(compare_df, use_container_width=True)

            elo_diff = float(profile_a["elo"]) - float(profile_b["elo"])
            if elo_diff > 0:
                st.info(f"{team_a} has the higher Elo rating by {elo_diff:.0f} points.")
            elif elo_diff < 0:
                st.info(f"{team_b} has the higher Elo rating by {abs(elo_diff):.0f} points.")
            else:
                st.info("Both teams have the same Elo rating.")

            st.markdown("### Prediction Probabilities")
            st.dataframe(probs_df, use_container_width=True)
            st.bar_chart(probs_df.set_index("Outcome"))

# =========================================================
# TOURNAMENT SIMULATION PAGE
# =========================================================
else:
    st.header("Tournament Simulation Results")

    if not simulation_path.exists():
        st.warning("Simulation results file not found. Run tournament_simulation.py first.")
    else:
        sim_df = pd.read_csv(simulation_path)

        st.markdown("### Stage Probabilities")
        st.dataframe(sim_df, use_container_width=True)

        top_n = st.slider("Show top teams", min_value=5, max_value=20, value=10, step=1)
        sort_by = st.selectbox(
            "Sort by",
            [
                "champion_probability",
                "final_probability",
                "semifinal_probability",
                "quarterfinal_probability",
                "qualify_probability"
            ],
            index=0
        )

        view_df = sim_df.sort_values(sort_by, ascending=False).head(top_n)

        st.markdown(f"### Top {top_n} Teams by {sort_by}")
        st.dataframe(view_df, use_container_width=True)

        chart_df = view_df.set_index("team")[[sort_by]]
        st.bar_chart(chart_df)

    if groups_path.exists():
        st.markdown("### Seeded Groups")
        groups_df = pd.read_csv(groups_path)
        st.dataframe(groups_df, use_container_width=True)