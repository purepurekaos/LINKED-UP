"""Turn raw nflreadpy tables into a leakage-safe, game-level feature matrix."""

import numpy as np
import pandas as pd

ROLL_WINDOW = 10  # trailing games used for "current form" per team


def _offensive_epa_per_play(team_stats: pd.DataFrame) -> pd.DataFrame:
    df = team_stats.copy()
    plays = df["attempts"] + df["carries"] + df["sacks_suffered"]
    off_epa = df["passing_epa"] + df["rushing_epa"] + df["receiving_epa"]
    df["off_epa_pp"] = np.where(plays > 0, off_epa / plays, np.nan)
    return df[["season", "week", "game_id", "team", "opponent_team", "off_epa_pp"]]


def build_team_games(schedules: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """One row per team per completed REG game, with trailing (pre-game) form features."""
    reg = schedules[schedules["game_type"] == "REG"].copy()
    played = reg.dropna(subset=["result"])

    stats = _offensive_epa_per_play(team_stats)

    # defense-allowed = the opponent's offensive epa/play in that same game
    opp_stats = stats.rename(columns={"team": "opponent_team", "opponent_team": "team", "off_epa_pp": "def_epa_pp"})
    stats = stats.merge(opp_stats, on=["season", "week", "game_id", "team", "opponent_team"], how="left")

    home = played[["game_id", "season", "week", "gameday", "home_team", "away_team",
                   "home_score", "away_score", "home_rest", "away_rest", "div_game"]].rename(
        columns={"home_team": "team", "away_team": "opponent_team",
                 "home_score": "points_for", "away_score": "points_against",
                 "home_rest": "rest", "away_rest": "opp_rest"})
    home["is_home"] = 1

    away = played[["game_id", "season", "week", "gameday", "home_team", "away_team",
                   "home_score", "away_score", "home_rest", "away_rest", "div_game"]].rename(
        columns={"away_team": "team", "home_team": "opponent_team",
                 "away_score": "points_for", "home_score": "points_against",
                 "away_rest": "rest", "home_rest": "opp_rest"})
    away["is_home"] = 0

    team_games = pd.concat([home, away], ignore_index=True)
    team_games["win"] = (team_games["points_for"] > team_games["points_against"]).astype(float)
    tie = team_games["points_for"] == team_games["points_against"]
    team_games.loc[tie, "win"] = 0.5

    team_games = team_games.merge(stats[["game_id", "team", "off_epa_pp", "def_epa_pp"]],
                                   on=["game_id", "team"], how="left")

    team_games = team_games.sort_values(["team", "season", "week"]).reset_index(drop=True)

    grp = team_games.groupby("team", group_keys=False)
    # pre-game trailing form: average over the last ROLL_WINDOW games, EXCLUDING the current one
    team_games["off_epa_form"] = grp["off_epa_pp"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())
    team_games["def_epa_form"] = grp["def_epa_pp"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())
    team_games["win_pct_form"] = grp["win"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())

    # current-form (not shifted): "as of right now", used to predict the NEXT, not-yet-played game
    team_games["off_epa_current"] = grp["off_epa_pp"].apply(
        lambda s: s.rolling(ROLL_WINDOW, min_periods=1).mean())
    team_games["def_epa_current"] = grp["def_epa_pp"].apply(
        lambda s: s.rolling(ROLL_WINDOW, min_periods=1).mean())
    team_games["win_pct_current"] = grp["win"].apply(
        lambda s: s.rolling(ROLL_WINDOW, min_periods=1).mean())

    return team_games


FEATURE_COLUMNS = ["net_epa_diff", "win_pct_diff", "rest_diff", "div_game"]


def _net_epa_diff(home_off, home_def, away_off, away_def):
    home_net = home_off - away_def
    away_net = away_off - home_def
    return home_net - away_net


def build_training_set(team_games: pd.DataFrame):
    """Game-level feature matrix (X, y) for every completed REG game with enough history."""
    home = team_games[team_games["is_home"] == 1].copy()

    away_cols = ["game_id", "off_epa_form", "def_epa_form", "win_pct_form", "rest"]
    away = team_games[team_games["is_home"] == 0][away_cols].rename(
        columns={"off_epa_form": "away_off_epa_form", "def_epa_form": "away_def_epa_form",
                 "win_pct_form": "away_win_pct_form", "rest": "away_rest"})

    games = home.merge(away, on="game_id", how="inner")
    games = games.rename(columns={"off_epa_form": "home_off_epa_form", "def_epa_form": "home_def_epa_form",
                                   "win_pct_form": "home_win_pct_form", "rest": "home_rest"})
    games = games[games["win"] != 0.5]  # drop ties (classification target only)

    games["net_epa_diff"] = _net_epa_diff(games["home_off_epa_form"], games["home_def_epa_form"],
                                           games["away_off_epa_form"], games["away_def_epa_form"])
    games["win_pct_diff"] = games["home_win_pct_form"] - games["away_win_pct_form"]
    games["rest_diff"] = games["home_rest"] - games["away_rest"]

    X = games[FEATURE_COLUMNS]
    y = games["win"].astype(int)
    meta = games[["game_id", "season", "week", "team", "opponent_team", "points_for", "points_against"]].rename(
        columns={"team": "home_team", "opponent_team": "away_team",
                 "points_for": "home_score", "points_against": "away_score"})
    return X, y, meta


def get_current_ratings(team_games: pd.DataFrame) -> pd.DataFrame:
    """Latest known form per team, to use as the pre-game state for a future matchup."""
    latest = team_games.sort_values(["team", "season", "week"]).groupby("team").tail(1)
    return latest[["team", "season", "week", "off_epa_current", "def_epa_current", "win_pct_current"]].rename(
        columns={"off_epa_current": "off_epa_form", "def_epa_current": "def_epa_form",
                 "win_pct_current": "win_pct_form"}).reset_index(drop=True)


def build_prediction_features(upcoming: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """upcoming: schedule rows (unplayed). Returns upcoming + feature columns, ready for model.predict."""
    df = upcoming.merge(ratings.add_prefix("home_"), left_on="home_team", right_on="home_team", how="left")
    df = df.merge(ratings.add_prefix("away_"), left_on="away_team", right_on="away_team", how="left")

    df["net_epa_diff"] = _net_epa_diff(df["home_off_epa_form"], df["home_def_epa_form"],
                                        df["away_off_epa_form"], df["away_def_epa_form"])
    df["win_pct_diff"] = df["home_win_pct_form"] - df["away_win_pct_form"]
    df["rest_diff"] = df["home_rest"] - df["away_rest"]
    return df
