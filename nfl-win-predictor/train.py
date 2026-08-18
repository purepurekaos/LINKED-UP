"""Train the win-probability model on all completed seasons and save it to models/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data import latest_completed_season
from model import train_and_save

if __name__ == "__main__":
    season = latest_completed_season()
    print(f"Training on seasons through {season}...")
    metrics = train_and_save(season)
    print("Done. Holdout evaluation:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
