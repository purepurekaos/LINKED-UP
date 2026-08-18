"""Reshape a season's schedule into a team-by-week grid, for a full-season overview."""

import pandas as pd

from data import load_schedules


def build_schedule_grid(season: int) -> pd.DataFrame:
    """One row per team, one column per REG week. Each cell is a dict:
    {opponent, is_home, result} where result is 'W'/'L'/'T'/None (not yet played), or
    None entirely for a bye week."""
    sched = load_schedules([season])
    reg = sched[sched["game_type"] == "REG"]
    if reg.empty:
        return pd.DataFrame()

    weeks = sorted(reg["week"].unique())
    teams = sorted(set(reg["home_team"]) | set(reg["away_team"]))

    home_games = reg[["week", "home_team", "away_team", "home_score", "away_score", "result"]].rename(
        columns={"home_team": "team", "away_team": "opponent", "home_score": "team_score",
                 "away_score": "opp_score"})
    home_games["is_home"] = True

    away_games = reg[["week", "away_team", "home_team", "away_score", "home_score", "result"]].rename(
        columns={"away_team": "team", "home_team": "opponent", "away_score": "team_score",
                 "home_score": "opp_score"})
    away_games["is_home"] = False

    team_games = pd.concat([home_games, away_games], ignore_index=True)

    def _cell(row):
        result = None
        if pd.notna(row["result"]):
            if row["team_score"] > row["opp_score"]:
                result = "W"
            elif row["team_score"] < row["opp_score"]:
                result = "L"
            else:
                result = "T"
        return {"opponent": row["opponent"], "is_home": row["is_home"], "result": result}

    team_games["cell"] = team_games.apply(_cell, axis=1)
    lookup = {(r["team"], r["week"]): r["cell"] for r in team_games.to_dict("records")}

    data = {team: {week: lookup.get((team, week)) for week in weeks} for team in teams}
    return pd.DataFrame.from_dict(data, orient="index", columns=weeks)
