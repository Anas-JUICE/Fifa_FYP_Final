from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from utils.helpers import load_dataset, save_team_profiles

def main():
    df = load_dataset()
    profiles_df = save_team_profiles(df)
    print("=== TEAM PROFILES BUILT ===")
    print(f"Teams saved: {len(profiles_df)}")
    print(profiles_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
