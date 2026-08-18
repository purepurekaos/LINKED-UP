"""Train and evaluate the win-probability model."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import FIRST_SEASON, load_schedules, load_team_stats
from features import build_team_games, build_training_set, get_current_ratings

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "win_probability.joblib"
HOLDOUT_SEASONS = 2  # most recent N completed seasons held out for evaluation


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000)),
    ])


def load_dataset(through_season: int):
    seasons = list(range(FIRST_SEASON, through_season + 1))
    sched = load_schedules(seasons)
    ts = load_team_stats(seasons)
    team_games = build_team_games(sched, ts)
    X, y, meta = build_training_set(team_games)
    valid = X.dropna().index
    return X.loc[valid], y.loc[valid], meta.loc[valid], team_games


def evaluate(X, y, meta):
    last_season = meta["season"].max()
    cutoff = last_season - HOLDOUT_SEASONS
    train_idx = meta["season"] <= cutoff
    test_idx = meta["season"] > cutoff

    pipe = make_pipeline()
    pipe.fit(X[train_idx], y[train_idx])
    proba = pipe.predict_proba(X[test_idx])[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "train_seasons": f"{meta['season'].min()}-{cutoff}",
        "test_seasons": f"{cutoff + 1}-{last_season}",
        "n_train": int(train_idx.sum()),
        "n_test": int(test_idx.sum()),
        "accuracy": accuracy_score(y[test_idx], preds),
        "log_loss": log_loss(y[test_idx], proba),
        "brier_score": brier_score_loss(y[test_idx], proba),
        "roc_auc": roc_auc_score(y[test_idx], proba),
        "baseline_home_win_rate": y[train_idx].mean(),
    }
    return metrics


def train_and_save(through_season: int):
    X, y, meta, team_games = load_dataset(through_season)
    metrics = evaluate(X, y, meta)

    final_model = make_pipeline()
    final_model.fit(X, y)

    ratings = get_current_ratings(team_games)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({
        "model": final_model,
        "ratings": ratings,
        "feature_columns": list(X.columns),
        "trained_through_season": through_season,
        "metrics": metrics,
    }, MODEL_PATH)
    return metrics


def load_artifacts():
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    from data import latest_completed_season
    season = latest_completed_season()
    metrics = train_and_save(season)
    print("Trained on seasons through", season)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"  {k}: {v}")
