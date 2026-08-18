# NFL Weekly Win Predictor

Predicts win probability for upcoming NFL games, week by week, using historical
team form pulled live from [nflverse](https://github.com/nflverse) via `nflreadpy`.

## Stack

- **nflreadpy** — schedules and team-game stats (EPA, etc.)
- **pandas** — feature engineering
- **scikit-learn** — logistic regression win-probability model
- **seaborn / matplotlib** — matchup and power-ranking charts
- **streamlit** — the web app itself

## How it works

1. `src/data.py` pulls schedules and per-team-per-game stats for every season since 2006,
   normalizing relocated-franchise abbreviations (e.g. `STL`/`SD`/`OAK` -> `LA`/`LAC`/`LV`)
   so history joins cleanly.
2. `src/features.py` builds, per team, a rolling ("last 10 games") trailing average of
   offensive EPA/play, defensive EPA/play allowed, and win %, computed **only from games
   before the one being predicted** (no leakage). Game-level features are then the home/away
   differential of these plus rest-day differential and a divisional-game flag.
3. `src/model.py` trains a logistic regression on standardized features, evaluated on the
   most recent two completed seasons held out from training, then refit on all data for
   live use. Artifacts (model + each team's current form + metrics) are saved to
   `models/win_probability.joblib`.
4. `src/predict.py` scores the next unplayed week of a season (`predict_next_week`), and can also
   backtest any already-played week (`season_backtest`) — reapplying the model to each game's actual
   pre-game rolling form, so you can see what it would have called versus what really happened.
5. `app.py` is the Streamlit front end: a Week selector spans every played week of a season plus the
   upcoming one, with team-logo matchup cards (win probabilities for the upcoming week, or a
   call-it-right/missed badge and running season record for played weeks), a confidence-ranked bar
   chart, and a league-wide offense-vs-defense scatter plot using each team's logo as its marker.

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the model

```bash
python train.py
```

This downloads/refreshes data as needed (cached to `cache/`), prints holdout evaluation
metrics, and saves the trained model to `models/win_probability.joblib`.

## Run the app

```bash
streamlit run app.py
```

The app will prompt you to train a model on first run if `models/win_probability.joblib`
doesn't exist yet, and has a "Retrain on latest data" button in the sidebar to pull in
newly completed games each week during the season.

## Current holdout performance

~63% accuracy / 0.69 ROC AUC on the last two completed seasons, vs. a 55.9% baseline for
always picking the home team. Roughly in line with public EPA-based NFL win models.
