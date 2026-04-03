from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from utils.helpers import predict_match_proba

def main():
    if len(sys.argv) < 3:
        print('Usage: python src/predict/predict_match.py "Brazil" "France"')
        return

    team_a = sys.argv[1]
    team_b = sys.argv[2]

    result = predict_match_proba(
        team_a=team_a,
        team_b=team_b,
        model_name="best",
        tournament="FIFA World Cup",
        neutral=True,
        match_importance=5.0
    )

    confidence = max(result["team_a_win"], result["draw"], result["team_b_win"])

    print("\n=== MATCH PREDICTION ===")
    print(f"Model      : {result['model_name']}")
    print(f"Match      : {result['team_a']} vs {result['team_b']}")
    print(f"Prediction : {result['predicted_outcome']}")
    print(f"Confidence : {confidence:.4f}")
    print(f"Team A Win : {result['team_a_win']:.4f}")
    print(f"Draw       : {result['draw']:.4f}")
    print(f"Team B Win : {result['team_b_win']:.4f}")

if __name__ == "__main__":
    main()
