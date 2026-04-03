from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from models.training_utils import train_and_save_model

def main():
    train_and_save_model("random_forest")

if __name__ == "__main__":
    main()
