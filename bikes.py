"""Shared helpers: load the data, build features, split by time, set up MLflow.

The scripts 1_ to 4_ import this file. The only part you may want to change is the
"mine" feature set in make_features (optional, for when you have time).
"""

import os
from pathlib import Path

import mlflow
import pandas as pd

TARGET = "count"  # bikes rented in that hour

# Time split. Months are numbered 1..24 (Jan 2011 = 1, Dec 2012 = 24).
# "Today" is the end of June 2012 (month 18).
TRAIN_MONTHS = range(1, 16)        # Jan 2011 - Mar 2012: learn from this
VALIDATION_MONTHS = range(16, 19)  # Apr - Jun 2012: compare models on this
NEW_MONTH = 19                     # Jul 2012: arrives later, in script 4

MODEL_NAME_LOCAL = "bike_demand"
MODEL_NAME_DATABRICKS = "workspace.default.bike_demand"


def repo_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:  # some Databricks run modes have no __file__
        return Path.cwd()


def on_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def load_bikes() -> pd.DataFrame:
    bikes = pd.read_csv(repo_dir() / "data" / "bike_sharing_hourly.csv")
    bikes["month_number"] = bikes["year"] * 12 + bikes["month"]
    return bikes


def make_features(bikes: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Turn raw columns into model inputs.

    raw:      every column as it is; text columns become 0/1 columns
    hour_onehot: also treat each hour as its own category, separately for
                 working days and days off (commuter peaks only happen on working days)
    mine:     your own idea, see below
    """
    raw = bikes[
        ["season", "year", "month", "hour", "holiday", "weekday", "workingday",
         "weather", "temp", "feel_temp", "humidity", "windspeed"]
    ].copy()
    features = pd.get_dummies(raw, columns=["season", "holiday", "workingday", "weather"], dtype=float)

    if feature_set == "raw":
        return features
    if feature_set == "hour_onehot":
        day_type = bikes["workingday"].astype(str).map({"True": "work", "False": "off"})
        hour_by_day_type = pd.get_dummies("h" + bikes["hour"].astype(str) + "_" + day_type, dtype=float)
        return pd.concat([features, hour_by_day_type], axis="columns")
    if feature_set == "mine":
        # OPTIONAL: add your own columns to `features`, then set FEATURES = "mine" in 1_train.py.
        # Ideas: a 0/1 column for rush hour (7-9 and 16-19 on working days),
        # temperature squared, "is it raining", or start from hour_onehot and add to it.
        # Example: features["is_rush_hour"] = bikes["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(float)
        return features
    raise ValueError(f"unknown feature_set: {feature_set}")


def split(bikes: pd.DataFrame, months: range | list[int]) -> pd.DataFrame:
    return bikes.loc[bikes["month_number"].isin(list(months))].copy()


def setup_mlflow() -> str:
    """Point MLflow at the right place and return the model registry name."""
    if on_databricks():
        from databricks.sdk import WorkspaceClient

        user = WorkspaceClient().current_user.me().user_name
        mlflow.set_registry_uri("databricks-uc")
        mlflow.set_experiment(f"/Users/{user}/bike-demand")
        return MODEL_NAME_DATABRICKS

    # On a laptop: keep everything in a local file next to the repo.
    mlflow.set_tracking_uri(f"sqlite:///{repo_dir() / 'mlflow.db'}")
    mlflow.set_experiment("bike-demand")
    return MODEL_NAME_LOCAL


def build_model(model: str, n_trees: int = 100, max_depth: int = 10):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    if model == "linear":
        return LinearRegression()
    if model == "forest":
        return RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, n_jobs=-1, random_state=42)
    raise ValueError(f"unknown model: {model}")
