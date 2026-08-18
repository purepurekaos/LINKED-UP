"""Generate win-probability predictions, and backtest the model against a season already in progress."""

import numpy as np
import pandas as pd

from data import FIRST_SEASON, load_schedules, load_team_stats, next_unplayed_week
from features import FEATURE_COLUMNS, build_prediction_features, build_team_games, build_training_set
from model import load_artifacts


def predict_next_week(season: int) -> pd.DataFrame:
    """Live prediction for the next not-yet-played week of a season."""
    artifacts = load_artifacts()
    upcoming = next_unplayed_week(season)
    if upcoming.empty:
        return upcoming

    df = build_prediction_features(upcoming, artifacts["ratings"])
    X = df[FEATURE_COLUMNS]
    known = X.dropna().index

    df["home_win_prob"] = pd.NA
    df.loc[known, "home_win_prob"] = artifacts["model"].predict_proba(X.loc[known])[:, 1]
    df["home_win_prob"] = df["home_win_prob"].astype(float)
    df["predicted_winner"] = df.apply(
        lambda r: r["home_team"] if pd.notna(r["home_win_prob"]) and r["home_win_prob"] >= 0.5
        else (r["away_team"] if pd.notna(r["home_win_prob"]) else None), axis=1)

    cols = ["season", "week", "gameday", "away_team", "home_team", "home_win_prob", "predicted_winner"]
    return df[cols].sort_values("gameday").reset_index(drop=True)


def season_backtest(season: int) -> pd.DataFrame:
    """Every completed REG game this season: what the model would have predicted beforehand
    (using only pre-game rolling form, same as training) versus what actually happened."""
    artifacts = load_artifacts()
    seasons = list(range(FIRST_SEASON, season + 1))
    sched = load_schedules(seasons)
    team_stats = load_team_stats(seasons)
    team_games = build_team_games(sched, team_stats)
    X, y, meta = build_training_set(team_games)

    mask = meta["season"] == season
    X, y, meta = X.loc[mask], y.loc[mask], meta.loc[mask]
    known = X.dropna().index
    if known.empty:
        return meta.iloc[0:0]

    out = meta.loc[known].copy()
    out["home_win_prob"] = artifacts["model"].predict_proba(X.loc[known])[:, 1]
    out["actual_home_win"] = y.loc[known].astype(bool).to_numpy()
    out["predicted_winner"] = np.where(out["home_win_prob"] >= 0.5, out["home_team"], out["away_team"])
    out["actual_winner"] = np.where(out["actual_home_win"], out["home_team"], out["away_team"])
    out["correct"] = out["predicted_winner"] == out["actual_winner"]

    gameday = sched[["game_id", "gameday"]]
    return out.merge(gameday, on="game_id", how="left").sort_values("gameday").reset_index(drop=True)
