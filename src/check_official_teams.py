import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # go back to project root

groups_path = ROOT / "data" / "official" / "worldcup_2026_groups.csv"
profiles_path = ROOT / "data" / "processed" / "team_profiles.csv"

groups_df = pd.read_csv(groups_path)
profiles_df = pd.read_csv(profiles_path)

official_teams = sorted(groups_df["team"].unique())
known_teams = set(profiles_df["team"].unique())

missing = [team for team in official_teams if team not in known_teams]

print("=== MISSING TEAMS ===")
for team in missing:
    print(team)

print(f"\nTotal missing: {len(missing)}")