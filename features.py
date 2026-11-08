"""Shared feature preparation for the NBA rookie-longevity project.

Both the training script and the Streamlit app import `prepare_features`, so a
saved model always sees exactly the same columns it was trained on.
"""
import numpy as np
import pandas as pd

TARGET_COL = "TARGET_5Yrs"

# Raw per-game rookie-season statistics (19 numeric inputs).
RAW_FEATURES = [
    "GP", "MIN", "PTS", "FGM", "FGA", "FG%", "3P Made", "3PA", "3P%",
    "FTM", "FTA", "FT%", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV",
]

# Columns created in this project (rates / ratios built from the raw stats).
ENGINEERED_FEATURES = [
    "pts_per_min", "reb_per_min", "ast_tov_ratio", "three_attempt_share", "ft_attempt_rate",
]

MODEL_INPUT_COLUMNS = RAW_FEATURES + ENGINEERED_FEATURES


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model-ready feature table from a raw dataframe.

    - keeps only the 19 raw stats (Name / target are ignored if present)
    - 3P% is missing only when a player never attempted a three, so it is set to 0
    - adds five engineered rate/ratio columns
    """
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    out = df[RAW_FEATURES].astype(float).copy()
    out["3P%"] = out["3P%"].fillna(0.0)

    minutes = out["MIN"].clip(lower=1.0)
    fga = out["FGA"].clip(lower=0.1)

    out["pts_per_min"] = out["PTS"] / minutes
    out["reb_per_min"] = out["REB"] / minutes
    out["ast_tov_ratio"] = out["AST"] / out["TOV"].clip(lower=0.1)
    out["three_attempt_share"] = out["3PA"] / fga
    out["ft_attempt_rate"] = out["FTA"] / fga

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out[MODEL_INPUT_COLUMNS]
