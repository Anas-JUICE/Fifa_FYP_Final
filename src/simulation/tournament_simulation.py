from __future__ import annotations

import itertools
import random
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from utils.helpers import RESULTS_DIR, FIGURES_DIR, load_team_profiles, predict_match_proba

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

@lru_cache(maxsize=None)
def cached_match_probs(team_a: str, team_b: str):
    result = predict_match_proba(
        team_a=team_a,
        team_b=team_b,
        model_name="best",
        tournament="FIFA World Cup",
        neutral=True,
        match_importance=5.0
    )
    return result["team_a_win"], result["draw"], result["team_b_win"]

def seed_48_teams_by_elo(profiles_df: pd.DataFrame):
    top_48 = profiles_df.sort_values("elo", ascending=False).head(48)["team"].tolist()
    groups = {chr(ord("A") + i): [] for i in range(12)}

    idx = 0
    direction = 1
    group_keys = list(groups.keys())

    for team in top_48:
        groups[group_keys[idx]].append(team)

        if direction == 1:
            if idx == len(group_keys) - 1:
                direction = -1
            else:
                idx += 1
        else:
            if idx == 0:
                direction = 1
            else:
                idx -= 1

    return groups

def sample_score_from_probs(p_a: float, p_d: float, p_b: float):
    outcome = np.random.choice(["A", "D", "B"], p=[p_a, p_d, p_b])

    if outcome == "D":
        score = random.choices([(0, 0), (1, 1), (2, 2)], weights=[0.50, 0.40, 0.10], k=1)[0]
        return score[0], score[1], "D"

    if outcome == "A":
        margin = random.choices([1, 2, 3], weights=[0.65, 0.25, 0.10], k=1)[0]
        goals_b = random.choices([0, 1, 2], weights=[0.55, 0.35, 0.10], k=1)[0]
        goals_a = goals_b + margin
        return goals_a, goals_b, "A"

    margin = random.choices([1, 2, 3], weights=[0.65, 0.25, 0.10], k=1)[0]
    goals_a = random.choices([0, 1, 2], weights=[0.55, 0.35, 0.10], k=1)[0]
    goals_b = goals_a + margin
    return goals_a, goals_b, "B"

def simulate_group(groups: dict):
    group_results = {}

    for group_name, teams in groups.items():
        table = {
            team: {
                "team": team,
                "group": group_name,
                "pts": 0,
                "gf": 0,
                "ga": 0,
                "gd": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0
            }
            for team in teams
        }

        for team_a, team_b in itertools.combinations(teams, 2):
            p_a, p_d, p_b = cached_match_probs(team_a, team_b)
            ga, gb, outcome = sample_score_from_probs(p_a, p_d, p_b)

            table[team_a]["gf"] += ga
            table[team_a]["ga"] += gb
            table[team_a]["gd"] = table[team_a]["gf"] - table[team_a]["ga"]

            table[team_b]["gf"] += gb
            table[team_b]["ga"] += ga
            table[team_b]["gd"] = table[team_b]["gf"] - table[team_b]["ga"]

            if outcome == "A":
                table[team_a]["pts"] += 3
                table[team_a]["wins"] += 1
                table[team_b]["losses"] += 1
            elif outcome == "B":
                table[team_b]["pts"] += 3
                table[team_b]["wins"] += 1
                table[team_a]["losses"] += 1
            else:
                table[team_a]["pts"] += 1
                table[team_b]["pts"] += 1
                table[team_a]["draws"] += 1
                table[team_b]["draws"] += 1

        standings = pd.DataFrame(list(table.values()))
        standings = standings.sort_values(
            ["pts", "gd", "gf", "wins", "team"],
            ascending=[False, False, False, False, True]
        ).reset_index(drop=True)
        standings["position"] = standings.index + 1
        group_results[group_name] = standings

    return group_results

def collect_qualifiers(group_results: dict):
    first_second = []
    third_place = []

    for _, standings in group_results.items():
        first_second.extend(standings.iloc[:2].to_dict("records"))
        third_place.append(standings.iloc[2].to_dict())

    third_df = pd.DataFrame(third_place).sort_values(
        ["pts", "gd", "gf", "wins", "team"],
        ascending=[False, False, False, False, True]
    ).reset_index(drop=True)

    best_thirds = third_df.head(8).to_dict("records")
    qualifiers = pd.DataFrame(first_second + best_thirds)

    qualifiers = qualifiers.sort_values(
        ["pts", "gd", "gf", "wins", "team"],
        ascending=[False, False, False, False, True]
    ).reset_index(drop=True)

    return qualifiers, third_df

def knockout_winner(team_a: str, team_b: str):
    p_a, p_d, p_b = cached_match_probs(team_a, team_b)
    ga, gb, outcome = sample_score_from_probs(p_a, p_d, p_b)

    if outcome == "A":
        return team_a
    if outcome == "B":
        return team_b

    no_draw_total = p_a + p_b
    a_prob = p_a / no_draw_total if no_draw_total > 0 else 0.5
    return np.random.choice([team_a, team_b], p=[a_prob, 1 - a_prob])

def pair_best_vs_worst(teams):
    teams = list(teams)
    pairings = []
    for i in range(len(teams) // 2):
        pairings.append((teams[i], teams[-(i + 1)]))
    return pairings

def play_round(teams):
    winners = []
    for team_a, team_b in pair_best_vs_worst(teams):
        winners.append(knockout_winner(team_a, team_b))
    return winners

def simulate_knockout(qualifiers_df: pd.DataFrame):
    round_of_32 = qualifiers_df["team"].tolist()
    round_of_16 = play_round(round_of_32)
    quarterfinalists = play_round(round_of_16)
    semifinalists = play_round(quarterfinalists)
    finalists = play_round(semifinalists)
    champion = play_round(finalists)[0]

    return {
        "round_of_16": set(round_of_16),
        "quarterfinalists": set(quarterfinalists),
        "semifinalists": set(semifinalists),
        "finalists": set(finalists),
        "champion": champion
    }

def run_simulation(iterations: int = 10000):
    profiles_df = load_team_profiles()
    groups = seed_48_teams_by_elo(profiles_df)

    qualify_counts = Counter()
    qf_counts = Counter()
    sf_counts = Counter()
    final_counts = Counter()
    champion_counts = Counter()

    for _ in range(iterations):
        group_results = simulate_group(groups)
        qualifiers, _ = collect_qualifiers(group_results)

        for team in qualifiers["team"].tolist():
            qualify_counts[team] += 1

        knockout = simulate_knockout(qualifiers)

        for team in knockout["quarterfinalists"]:
            qf_counts[team] += 1
        for team in knockout["semifinalists"]:
            sf_counts[team] += 1
        for team in knockout["finalists"]:
            final_counts[team] += 1
        champion_counts[knockout["champion"]] += 1

    top_48 = profiles_df.sort_values("elo", ascending=False).head(48)["team"].tolist()

    result = pd.DataFrame({
        "team": top_48,
        "qualify_probability": [qualify_counts[t] / iterations for t in top_48],
        "quarterfinal_probability": [qf_counts[t] / iterations for t in top_48],
        "semifinal_probability": [sf_counts[t] / iterations for t in top_48],
        "final_probability": [final_counts[t] / iterations for t in top_48],
        "champion_probability": [champion_counts[t] / iterations for t in top_48],
    }).sort_values(["champion_probability", "final_probability", "team"], ascending=[False, False, True]).reset_index(drop=True)

    result.to_csv(RESULTS_DIR / "worldcup_simulation_results.csv", index=False)

    seed_info = pd.DataFrame(
        [(group, ", ".join(teams)) for group, teams in groups.items()],
        columns=["group", "teams"]
    )
    seed_info.to_csv(RESULTS_DIR / "auto_seeded_groups.csv", index=False)

    plot_df = result.head(15).sort_values("champion_probability", ascending=True)
    plt.figure(figsize=(8, 6))
    plt.barh(plot_df["team"], plot_df["champion_probability"])
    plt.title("Top 15 Champion Probabilities")
    plt.xlabel("Probability")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "champion_probabilities_top15.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("=== TOURNAMENT SIMULATION COMPLETE ===")
    print(result.head(15).to_string(index=False))
    print(f"Saved stage probabilities to: {RESULTS_DIR / 'worldcup_simulation_results.csv'}")
    print(f"Saved seeded groups to     : {RESULTS_DIR / 'auto_seeded_groups.csv'}")

if __name__ == "__main__":
    run_simulation(iterations=10000)
