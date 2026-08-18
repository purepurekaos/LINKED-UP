"""Raw data access for NFL history via nflreadpy, returned as pandas DataFrames."""

import nflreadpy as nfl

nfl.config.update_config(cache_mode="filesystem")

FIRST_SEASON = 2006  # modern-era cutoff: post-2005 realignment, EPA data is reliable from here

# schedules keeps the abbreviation in use at the time; team_stats retroactively applies
# each franchise's current code across its whole history. Normalize to the latter.
_RELOCATED_TEAM_CODES = {"STL": "LA", "SD": "LAC", "OAK": "LV"}


def load_schedules(seasons):
    df = nfl.load_schedules(seasons).to_pandas()
    df["home_team"] = df["home_team"].replace(_RELOCATED_TEAM_CODES)
    df["away_team"] = df["away_team"].replace(_RELOCATED_TEAM_CODES)
    return df


def load_team_stats(seasons):
    """nflverse only publishes a season's stats file once games have actually been played —
    a not-yet-started season (e.g. the upcoming one, in the offseason) 404s. Drop it and retry once."""
    try:
        return nfl.load_team_stats(seasons).to_pandas()
    except Exception:
        trimmed = [s for s in seasons if s < max(seasons)]
        if not trimmed:
            raise
        return nfl.load_team_stats(trimmed).to_pandas()


def load_teams():
    return nfl.load_teams().to_pandas()


def current_season():
    return nfl.get_current_season()


def latest_completed_season() -> int:
    """Most recent season whose full regular season has been played out."""
    probe = load_schedules(list(range(current_season() - 1, current_season() + 1)))
    reg = probe[probe["game_type"] == "REG"]
    complete = reg.groupby("season")["result"].apply(lambda s: s.notna().all())
    return int(complete[complete].index.max())


def next_unplayed_week(season: int):
    """(season, week) schedule rows for the next REG week that hasn't been played yet."""
    sched = load_schedules([season])
    reg = sched[sched["game_type"] == "REG"]
    unplayed = reg[reg["result"].isna()]
    if unplayed.empty:
        return sched.iloc[0:0]
    next_week = unplayed["week"].min()
    return unplayed[unplayed["week"] == next_week]
