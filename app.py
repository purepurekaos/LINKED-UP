import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from data import current_season, latest_completed_season, load_teams
from features import FEATURE_COLUMNS, build_prediction_features
from logos import get_logo
from model import MODEL_PATH, load_artifacts, train_and_save
from predict import predict_next_week, season_backtest
from scheduling import build_schedule_grid

st.set_page_config(page_title="NFL Win Predictor", page_icon="🏈", layout="wide")

INK = "#F4F6FB"
INK_MUTED = "#94A3B8"
GRID = "#232C45"
BG_PAGE = "#0B1120"
BG_CARD = "#141B2E"
AFC_COLOR = "#e66767"
NFC_COLOR = "#3987e5"

def _hex_to_rgb(h):
    h = (h or "#4B5876").lstrip("#")
    if len(h) != 6:
        h = "4B5876"
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _luminance(h):
    r, g, b = _hex_to_rgb(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _color_dist(h1, h2):
    r1, g1, b1 = _hex_to_rgb(h1)
    r2, g2, b2 = _hex_to_rgb(h2)
    return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5


_BG_LUM = _luminance("#141B2E")
_LUM_FLOOR = 55


def _lighten_to_floor(hex_color, target=95):
    r, g, b = _hex_to_rgb(hex_color)
    for step in range(1, 21):
        t = step / 20
        nr, ng, nb = r + (255 - r) * t, g + (255 - g) * t, b + (255 - b) * t
        if 0.2126 * nr + 0.7152 * ng + 0.0722 * nb >= target:
            break
    return f"#{int(nr):02x}{int(ng):02x}{int(nb):02x}"


def pick_matchup_colors(away_primary, away_secondary, home_primary, home_secondary):
    """Choose each team's display color (primary or secondary) to maximize contrast
    against its opponent and the card background — two teams can share a primary hex
    (e.g. NE and SEA are both '#002244' in nflverse's data), and some teams' colors
    (e.g. Ravens purple/black) are too dark to read on a dark card at all."""
    candidates = [
        (away_primary, home_primary),
        (away_secondary, home_primary),
        (away_primary, home_secondary),
        (away_secondary, home_secondary),
    ]

    bright_enough = [pair for pair in candidates if all(_luminance(c) >= _LUM_FLOOR for c in pair)]
    pool = bright_enough if bright_enough else candidates
    away, home = max(pool, key=lambda pair: _color_dist(*pair))

    if _luminance(away) < _LUM_FLOOR:
        away = _lighten_to_floor(away)
    if _luminance(home) < _LUM_FLOOR:
        home = _lighten_to_floor(home)
    return away, home


plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "sans-serif"],
    "text.color": INK,
    "axes.labelcolor": INK_MUTED,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.edgecolor": GRID,
})


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

    .hero-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3.4rem;
        letter-spacing: 0.04em;
        line-height: 1;
        margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #F4F6FB 30%, #8B7FF5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1rem;
        max-width: 680px;
        line-height: 1.6;
        margin-bottom: 0.9rem;
    }
    .badge-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 1.6rem; }
    .badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 999px;
        background: rgba(109,93,242,0.14);
        color: #B3A8FF;
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        border: 1px solid rgba(139,127,245,0.3);
    }
    .badge.muted {
        background: rgba(255,255,255,0.04);
        color: #94A3B8;
        border-color: rgba(255,255,255,0.1);
    }

    .section-label {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.6rem;
        letter-spacing: 0.03em;
        color: #F4F6FB;
        margin: 0.2rem 0 0.1rem 0;
    }
    .section-desc { color: #94A3B8; font-size: 0.92rem; margin-bottom: 1.1rem; line-height: 1.5; }

    .matchup-card {
        background: #141B2E;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 18px;
        padding: 20px 26px;
        margin-bottom: 16px;
        transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
    }
    .matchup-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 28px rgba(0,0,0,0.38);
        border-color: rgba(139,127,245,0.35);
    }
    .matchup-row { display: flex; align-items: center; justify-content: space-between; gap: 18px; }
    .team-block { display: flex; align-items: center; gap: 13px; flex: 1; min-width: 0; }
    .team-block.home-block { flex-direction: row-reverse; text-align: right; }
    .team-logo { width: 42px; height: 42px; object-fit: contain; flex-shrink: 0; }
    .team-name { font-weight: 700; font-size: 1.08rem; letter-spacing: 0.01em; }
    .team-prob {
        color: #94A3B8; font-size: 0.82rem; font-variant-numeric: tabular-nums; margin-top: 1px;
    }
    .team-prob.leader { color: #B3A8FF; font-weight: 600; }
    .matchup-meta { text-align: center; min-width: 108px; flex-shrink: 0; }
    .matchup-meta .at-label {
        color: #6B7A99; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em;
    }
    .matchup-meta .date-label {
        font-size: 0.86rem; color: #CBD3E6; font-variant-numeric: tabular-nums; margin-top: 2px;
    }

    .prob-track {
        display: flex; height: 9px; border-radius: 999px; overflow: hidden;
        margin-top: 16px; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.12);
    }
    .prob-seg { height: 100%; }
    .prob-seg.away-seg { margin-right: 2px; }
    .prob-seg.home-seg { margin-left: 2px; }

    .result-badge {
        display: inline-block; margin-top: 14px; padding: 5px 12px; border-radius: 8px;
        font-size: 0.78rem; font-weight: 600;
    }
    .result-badge.result-correct {
        background: rgba(52,211,153,0.12); color: #6EE7B7; border: 1px solid rgba(52,211,153,0.32);
    }
    .result-badge.result-wrong {
        background: rgba(248,113,113,0.12); color: #FCA5A5; border: 1px solid rgba(248,113,113,0.32);
    }

    .ad-slot {
        display: flex; align-items: center; justify-content: center; flex-direction: column; gap: 4px;
        border: 1px dashed rgba(255,255,255,0.14);
        border-radius: 14px;
        color: #6B7A99;
        background: rgba(255,255,255,0.015);
        margin: 20px 0;
        padding: 14px;
    }
    .ad-slot .ad-kicker {
        font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.12em; color: #6B7A99;
    }
    .ad-slot .ad-size { font-size: 0.72rem; color: #4B5876; }

    .stTabs [data-baseweb="tab"] {
        font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.82rem;
    }

    .schedule-wrap {
        overflow-x: auto; border: 1px solid rgba(255,255,255,0.08); border-radius: 14px;
        max-height: 78vh; overflow-y: auto;
    }
    .schedule-table { border-collapse: collapse; font-size: 0.76rem; white-space: nowrap; width: 100%; }
    .schedule-table th, .schedule-table td {
        padding: 7px 10px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .schedule-table thead th {
        position: sticky; top: 0; background: #141B2E; font-weight: 700; color: #94A3B8;
        text-transform: uppercase; font-size: 0.66rem; letter-spacing: 0.06em; z-index: 2;
    }
    .schedule-table thead th.team-header { left: 0; z-index: 4; text-align: left; }
    .schedule-table thead th.current-week { background: rgba(109,93,242,0.28); color: #F4F6FB; }
    .schedule-table td.team-cell, .schedule-table thead th.team-header {
        position: sticky; left: 0; background: #141B2E; text-align: left; z-index: 1; min-width: 148px;
    }
    .schedule-table tr.division-row td {
        background: #0B1120; color: #8593B3; font-size: 0.66rem; text-transform: uppercase;
        letter-spacing: 0.08em; padding: 7px 10px; font-weight: 700; text-align: left;
    }
    .schedule-table td.current-week-col { background: rgba(109,93,242,0.07); }
    .team-cell-inner { display: flex; align-items: center; gap: 8px; }
    .team-cell-inner img { width: 22px; height: 22px; object-fit: contain; }
    .opp-cell { border-radius: 7px; padding: 3px 7px; display: inline-block; min-width: 58px; font-weight: 600; }
    .opp-result { margin-left: 4px; font-weight: 800; }
    .opp-result-W { color: #6EE7B7; }
    .opp-result-L { color: #FCA5A5; }
    .opp-result-T { color: #FDE68A; }
    .opp-bye { color: #4B5876; font-style: italic; font-size: 0.7rem; }

    a { color: #B3A8FF; }

    @media (prefers-reduced-motion: reduce) {
        .matchup-card { transition: none; }
    }
    </style>
    """, unsafe_allow_html=True)


def ad_slot(label="Advertisement", size="300 × 250", height=140):
    st.markdown(f"""
    <div class="ad-slot" style="min-height:{height}px">
        <span class="ad-kicker">{label}</span>
        <span class="ad-size">{size}</span>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_artifacts(_cache_key: int):
    return load_artifacts()


@st.cache_data(show_spinner=False)
def get_teams():
    return load_teams()


@st.cache_data(show_spinner=False)
def get_predictions(season: int, _cache_key: int):
    return predict_next_week(season)


@st.cache_data(show_spinner=False)
def get_season_backtest(season: int, _cache_key: int):
    return season_backtest(season)


@st.cache_data(show_spinner=False)
def get_schedule_grid(season: int):
    return build_schedule_grid(season)


@st.cache_resource(show_spinner=False)
def get_logo_cached(team: str, url: str):
    return get_logo(team, url)


def train_model_now():
    season = latest_completed_season()
    with st.spinner(f"Training on seasons through {season} — pulling fresh nflverse data (~15s)..."):
        train_and_save(season)
    st.cache_resource.clear()
    st.cache_data.clear()


def format_gameday(value) -> str:
    try:
        return pd.to_datetime(value).strftime("%a, %b %-d")
    except Exception:
        return str(value)


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------

inject_css()

st.markdown('<div class="hero-title">NFL WIN PREDICTOR</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Week-by-week win probabilities from rolling team form — '
    'offensive &amp; defensive EPA per play, recent record, rest days, and divisional matchups. '
    'Live data via nflreadpy (nflverse).</div>',
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.info("No trained model found yet.")
    if st.button("Train model now", type="primary"):
        train_model_now()
        st.rerun()
    st.stop()

cache_key = MODEL_PATH.stat().st_mtime_ns
artifacts = get_artifacts(cache_key)
teams = get_teams()
metrics = artifacts["metrics"]
logo_map = dict(zip(teams["team_abbr"], teams["team_logo_espn"]))
color_map = dict(zip(teams["team_abbr"], teams["team_color"]))
color2_map = dict(zip(teams["team_abbr"], teams["team_color2"]))

with st.sidebar:
    st.markdown('<div class="section-label" style="font-size:1.15rem;">MODEL</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{metrics['accuracy']:.1%}", help=f"Held out: {metrics['test_seasons']}")
    c2.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
    st.caption(f"Trained through {artifacts['trained_through_season']} · home-favorite baseline "
               f"{metrics['baseline_home_win_rate']:.1%}")
    if st.button("Retrain on latest data", width='stretch'):
        train_model_now()
        st.rerun()

    st.divider()
    season = st.number_input("Season", min_value=2020, max_value=current_season() + 1,
                              value=current_season() + 1 if latest_completed_season() == current_season()
                              else current_season())

    backtest_df = get_season_backtest(int(season), cache_key)
    predictions = get_predictions(int(season), cache_key)

    played_weeks = sorted(backtest_df["week"].unique().tolist()) if not backtest_df.empty else []
    upcoming_week = int(predictions["week"].iloc[0]) if not predictions.empty else None

    week_labels = {w: f"Week {w}" for w in played_weeks}
    if upcoming_week is not None:
        week_labels[upcoming_week] = f"Week {upcoming_week} — Upcoming"
    week_choices = list(week_labels.keys())

    if week_choices:
        selected_week = st.selectbox("Week", options=week_choices, index=len(week_choices) - 1,
                                      format_func=lambda w: week_labels[w])
    else:
        selected_week = None
        st.caption("No games found for this season yet.")

    if not backtest_df.empty:
        season_correct = int(backtest_df["correct"].sum())
        season_total = len(backtest_df)
        st.metric("Season record", f"{season_correct}-{season_total - season_correct}",
                  help=f"Retrospective: what the model would have called each week this season, "
                       f"{season_correct / season_total:.1%} correct. Not the held-out accuracy above — "
                       f"this season's games were part of the training data.")

    st.divider()
    ad_slot(size="300 × 250", height=250)
    st.caption("Built with nflreadpy · pandas · scikit-learn · seaborn · Streamlit")


def render_matchup_card(away_team, home_team, gameday, home_prob, actual_winner=None, i=None):
    home_logo = logo_map.get(home_team, "")
    away_logo = logo_map.get(away_team, "")
    away_color, home_color = pick_matchup_colors(
        color_map.get(away_team, "#4B5876"), color2_map.get(away_team, "#4B5876"),
        color_map.get(home_team, "#4B5876"), color2_map.get(home_team, "#4B5876"))

    has_prob = pd.notna(home_prob)
    away_prob = 1 - home_prob if has_prob else None
    away_prob_html = f"{away_prob:.0%} win" if has_prob else "N/A"
    home_prob_html = f"{home_prob:.0%} win" if has_prob else "N/A"
    away_cls = "team-prob leader" if has_prob and away_prob >= 0.5 else "team-prob"
    home_cls = "team-prob leader" if has_prob and home_prob >= 0.5 else "team-prob"

    bar_html = ""
    if has_prob:
        bar_html = (
            '<div class="prob-track">'
            f'<div class="prob-seg away-seg" style="width:{away_prob*100:.2f}%; background:{away_color};"></div>'
            f'<div class="prob-seg home-seg" style="width:{home_prob*100:.2f}%; background:{home_color};"></div>'
            '</div>'
        )

    result_html = ""
    if actual_winner is not None and has_prob:
        predicted_winner = home_team if home_prob >= 0.5 else away_team
        correct = predicted_winner == actual_winner
        badge_cls = "result-correct" if correct else "result-wrong"
        badge_text = "Model called it" if correct else "Model missed"
        result_html = (f'<div class="result-badge {badge_cls}">{badge_text} — '
                        f'<strong>{actual_winner}</strong> won</div>')

    card_html = (
        '<div class="matchup-card"><div class="matchup-row">'
        '<div class="team-block">'
        f'<img class="team-logo" src="{away_logo}" alt="{away_team} logo">'
        f'<div><div class="team-name">{away_team}</div>'
        f'<div class="{away_cls}">{away_prob_html}</div></div>'
        '</div>'
        '<div class="matchup-meta">'
        '<div class="at-label">Road @ Home</div>'
        f'<div class="date-label">{format_gameday(gameday)}</div>'
        '</div>'
        '<div class="team-block home-block">'
        f'<img class="team-logo" src="{home_logo}" alt="{home_team} logo">'
        f'<div><div class="team-name">{home_team}</div>'
        f'<div class="{home_cls}">{home_prob_html}</div></div>'
        '</div>'
        '</div>'
        f'{bar_html}'
        f'{result_html}'
        '</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)
    if i == 5:
        ad_slot(label="Advertisement", size="728 × 90", height=90)


tab_week, tab_schedule, tab_rankings, tab_about = st.tabs(
    ["Matchups", "Full Schedule", "Power Rankings", "About the Model"])

# ----------------------------------------------------------------------------
# Tab: Matchups
# ----------------------------------------------------------------------------
with tab_week:
    if selected_week is None:
        st.warning("No games found for this season — either it hasn't been scheduled yet, "
                   "or the season lookup failed.")
    elif selected_week == upcoming_week:
        st.markdown(f'<div class="badge-row">'
                    f'<span class="badge">{int(season)} Season</span>'
                    f'<span class="badge">Week {selected_week}</span>'
                    f'<span class="badge muted">{len(predictions)} games</span>'
                    f'<span class="badge muted">Live prediction</span>'
                    f'</div>', unsafe_allow_html=True)

        for i, row in predictions.iterrows():
            render_matchup_card(row["away_team"], row["home_team"], row["gameday"], row["home_win_prob"], i=i)

        chart_df = predictions.dropna(subset=["home_win_prob"]).copy()
        if not chart_df.empty:
            chart_df["matchup"] = chart_df["away_team"] + " @ " + chart_df["home_team"]
            chart_df = chart_df.sort_values("home_win_prob")

            st.markdown('<div class="section-label">Confidence board</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-desc">Every matchup this week, ranked by how lopsided the model '
                        'thinks it is. Blue leans home, red leans away.</div>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(9, max(3, 0.42 * len(chart_df))))
            fig.patch.set_facecolor(BG_PAGE)
            ax.set_facecolor(BG_PAGE)

            values = chart_df["home_win_prob"].to_numpy()
            colors = np.where(values >= 0.5, NFC_COLOR, AFC_COLOR)
            bar_lengths = np.abs(values - 0.5)
            ax.barh(chart_df["matchup"], bar_lengths, left=np.minimum(values, 0.5), color=colors, height=0.55)

            for y, (m, v) in enumerate(zip(chart_df["matchup"], values)):
                ax.text(v + (0.018 if v >= 0.5 else -0.018), y, f"{v:.0%}",
                        va="center", ha="left" if v >= 0.5 else "right", fontsize=9, color=INK, fontweight="bold")

            ax.axvline(0.5, color=INK_MUTED, linewidth=0.8, linestyle="--", alpha=0.6)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Home team win probability")
            for spine in ["top", "right", "left"]:
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.6)
            ax.set_axisbelow(True)
            fig.tight_layout()
            st.pyplot(fig, width='stretch')
    else:
        week_df = backtest_df[backtest_df["week"] == selected_week].sort_values("gameday")
        week_correct = int(week_df["correct"].sum())
        st.markdown(f'<div class="badge-row">'
                    f'<span class="badge">{int(season)} Season</span>'
                    f'<span class="badge">Week {selected_week}</span>'
                    f'<span class="badge muted">{len(week_df)} games</span>'
                    f'<span class="badge muted">{week_correct}-{len(week_df) - week_correct} vs. model pick</span>'
                    f'</div>', unsafe_allow_html=True)

        for i, row in week_df.reset_index(drop=True).iterrows():
            render_matchup_card(row["away_team"], row["home_team"], row["gameday"], row["home_win_prob"],
                                 actual_winner=row["actual_winner"], i=i)

# ----------------------------------------------------------------------------
# Tab: Full Schedule
# ----------------------------------------------------------------------------
with tab_schedule:
    st.markdown('<div class="section-label">Full season schedule</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Every team, every week — who they play, home (vs) or away (@), '
                'and the result once it\'s played. The highlighted column is whatever week is selected on '
                'the Matchups tab.</div>', unsafe_allow_html=True)

    grid = get_schedule_grid(int(season))
    if grid.empty:
        st.warning("No schedule found for this season.")
    else:
        weeks = list(grid.columns)
        team_info = teams.set_index("team_abbr")
        ordered_teams = [t for t in team_info.sort_values(["team_conf", "team_division"]).index if t in grid.index]

        header_cells = "".join(
            f'<th class="{"current-week" if w == selected_week else ""}">Wk {w}</th>' for w in weeks)
        header_html = f'<thead><tr><th class="team-header">Team</th>{header_cells}</tr></thead>'

        body_rows = []
        current_division = None
        for team in ordered_teams:
            info = team_info.loc[team]
            if info["team_division"] != current_division:
                current_division = info["team_division"]
                body_rows.append(
                    f'<tr class="division-row"><td colspan="{len(weeks) + 1}">{current_division}</td></tr>')

            logo = logo_map.get(team, "")
            cells = [f'<td class="team-cell"><div class="team-cell-inner">'
                     f'<img src="{logo}" alt="{team} logo"><span>{team}</span></div></td>']
            for w in weeks:
                cell = grid.loc[team, w]
                col_cls = "current-week-col" if w == selected_week else ""
                if cell is None:
                    cells.append(f'<td class="{col_cls}"><span class="opp-bye">BYE</span></td>')
                    continue
                opp = cell["opponent"]
                prefix = "vs" if cell["is_home"] else "@"
                opp_color = color_map.get(opp, "#4B5876")
                if _luminance(opp_color) < _LUM_FLOOR:
                    opp_color = _lighten_to_floor(opp_color)
                result_html = (f'<span class="opp-result opp-result-{cell["result"]}">{cell["result"]}</span>'
                                if cell["result"] else "")
                cells.append(
                    f'<td class="{col_cls}"><span class="opp-cell" '
                    f'style="background:{opp_color}22; color:{opp_color};">{prefix} {opp}</span>{result_html}</td>')
            body_rows.append(f'<tr>{"".join(cells)}</tr>')

        table_html = (f'<div class="schedule-wrap"><table class="schedule-table">{header_html}'
                      f'<tbody>{"".join(body_rows)}</tbody></table></div>')
        st.markdown(table_html, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Tab: Power Rankings
# ----------------------------------------------------------------------------
with tab_rankings:
    st.markdown('<div class="section-label">Team form entering next matchup</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Rolling average offensive / defensive EPA per play over each '
                "team's last 10 games. Top-right = strong offense and a strong defense.</div>",
                unsafe_allow_html=True)

    ratings = artifacts["ratings"].merge(teams[["team_abbr", "team_conf", "team_division"]],
                                          left_on="team", right_on="team_abbr", how="left")

    x_pad = (ratings["off_epa_form"].max() - ratings["off_epa_form"].min()) * 0.18
    y_pad = (ratings["def_epa_form"].max() - ratings["def_epa_form"].min()) * 0.18
    x_lim = (ratings["off_epa_form"].min() - x_pad, ratings["off_epa_form"].max() + x_pad)
    y_lim = (ratings["def_epa_form"].min() - y_pad, ratings["def_epa_form"].max() + y_pad)

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    fig2.patch.set_facecolor(BG_PAGE)
    ax2.set_facecolor(BG_PAGE)

    x_mean, y_mean = ratings["off_epa_form"].mean(), ratings["def_epa_form"].mean()
    ax2.axhline(y_mean, color=GRID, linewidth=1, linestyle=(0, (4, 3)))
    ax2.axvline(x_mean, color=GRID, linewidth=1, linestyle=(0, (4, 3)))

    for _, r in ratings.iterrows():
        img = get_logo_cached(r["team"], logo_map.get(r["team"], ""))
        if img is not None:
            im = OffsetImage(img, zoom=24 / max(img.size))
            ab = AnnotationBbox(im, (r["off_epa_form"], r["def_epa_form"]), frameon=False, zorder=3)
            ax2.add_artist(ab)
        else:
            conf_color = NFC_COLOR if r.get("team_conf") == "NFC" else AFC_COLOR
            ax2.scatter(r["off_epa_form"], r["def_epa_form"], color=conf_color, s=90, zorder=3)
            ax2.annotate(r["team"], (r["off_epa_form"], r["def_epa_form"]), fontsize=8, color=INK_MUTED,
                         xytext=(5, 5), textcoords="offset points")

    ax2.set_xlim(*x_lim)
    ax2.set_ylim(*y_lim)
    ax2.invert_yaxis()

    # axes-fraction coords: (0,0)=bottom-left..(1,1)=top-right of the VISIBLE plot,
    # independent of invert_yaxis() — avoids reasoning about flipped data coordinates.
    corner_kwargs = dict(fontsize=9, color=INK_MUTED, style="italic", transform=ax2.transAxes)
    ax2.text(0.99, 0.98, "elite both sides", ha="right", va="top", **corner_kwargs)
    ax2.text(0.01, 0.98, "strong D, weak O", ha="left", va="top", **corner_kwargs)
    ax2.text(0.99, 0.02, "weak D, strong O", ha="right", va="bottom", **corner_kwargs)
    ax2.text(0.01, 0.02, "weak D, weak O", ha="left", va="bottom", **corner_kwargs)

    ax2.set_xlabel("Offensive EPA / play  (better  →)")
    ax2.set_ylabel("Defensive EPA / play allowed  (lower is better)")
    for spine in ax2.spines.values():
        spine.set_color(GRID)
    ax2.grid(color=GRID, linewidth=0.6, alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.tick_params(colors=INK_MUTED)
    fig2.tight_layout()
    st.pyplot(fig2, width='stretch')

    with st.expander("View as table"):
        table = ratings[["team", "team_conf", "team_division", "off_epa_form", "win_pct_form"]].rename(
            columns={"team": "Team", "team_conf": "Conf", "team_division": "Division",
                     "off_epa_form": "Off EPA/play", "win_pct_form": "Win % (L10)"})
        table["Def EPA/play allowed"] = ratings["def_epa_form"]
        st.dataframe(table.sort_values("Off EPA/play", ascending=False), width='stretch', hide_index=True)

# ----------------------------------------------------------------------------
# Tab: About
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown('<div class="section-label">How this works</div>', unsafe_allow_html=True)
    st.markdown(f"""
- **Data** — `nflreadpy` pulls play-by-play-derived team stats and schedules from the nflverse project,
  covering every season back to 2006.
- **Features** — `{', '.join(FEATURE_COLUMNS)}`: net EPA/play differential (each team's offense against the
  opponent's defense, both directions), recent win % differential, rest-day differential, and a
  divisional-game flag.
- **Leakage control** — every rolling feature is computed only from games *before* the one being predicted.
- **Model** — logistic regression on standardized features, evaluated on a held-out set of the most recent
  completed seasons (unseen during training) before being refit on all available data for live predictions.
- **Backtest** — the Week selector isn't limited to the next kickoff: pick any already-played week to see
  what the model would have called beforehand (using only that game's pre-game rolling form) next to what
  actually happened, with a running season record in the sidebar. Note this is retrospective, not held-out —
  this season's games were part of the training data, so it reads optimistic next to the accuracy above.
- **Stack** — pandas for feature engineering, scikit-learn for the model, seaborn/matplotlib for the charts
  above, and Streamlit for this interface.
    """)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    m2.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
    m3.metric("Log loss", f"{metrics['log_loss']:.3f}")
    m4.metric("Brier score", f"{metrics['brier_score']:.3f}")
    st.caption(f"Trained on {metrics['train_seasons']} · held out on {metrics['test_seasons']} · "
               f"{metrics['n_train']} training games / {metrics['n_test']} test games.")

    st.divider()
    ad_slot(label="Advertisement", size="728 × 90", height=90)
