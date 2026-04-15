import pandas as pd
from pathlib import Path
import difflib

ROOT = Path(__file__).resolve().parents[1]

groups_path = ROOT / "data" / "official" / "worldcup_2026_groups.csv"
profiles_path = ROOT / "data" / "processed" / "team_profiles.csv"

groups_df = pd.read_csv(groups_path)
profiles_df = pd.read_csv(profiles_path)

official_teams = sorted(groups_df["team"].unique())
known_teams = sorted(profiles_df["team"].unique())

missing = [team for team in official_teams if team not in known_teams]

print("=== POSSIBLE MATCHES ===")
for team in missing:
    matches = difflib.get_close_matches(team, known_teams, n=5, cutoff=0.3)
    print(f"\n{team}  -->  {matches}")