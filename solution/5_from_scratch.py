"""Finished version of 5_from_scratch.py: a full MLflow run written by hand, no helpers.

Tunes one decision tree setting (max_depth) and records every try as its own run.
To run it from this folder, set EMAIL first.
"""

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

EMAIL = "you@example.com"  # the email you log in to Databricks with
FEATURES = ["hour", "weekday", "temp", "humidity", "windspeed"]

# 1. Say where the runs go: one experiment = one folder of runs.
mlflow.set_experiment(f"/Users/{EMAIL}/bike-scratch")

# 2. Load the data and split by time: learn from 2011, check on Jan-Jun 2012.
bikes = pd.read_csv("../data/bike_sharing_hourly.csv")
train = bikes.loc[bikes["year"] == 0]
validation = bikes.loc[(bikes["year"] == 1) & (bikes["month"] <= 6)]

# 3. One run per setting we try.
for depth in [3, 6, 10, 15]:
    with mlflow.start_run(run_name=f"tree-depth-{depth}"):
        mlflow.set_tag("written_by", "me")
        mlflow.log_param("max_depth", depth)
        mlflow.log_param("features", ", ".join(FEATURES))

        model = DecisionTreeRegressor(max_depth=depth, random_state=42)
        model.fit(train[FEATURES], train["count"])
        predictions = model.predict(validation[FEATURES])

        mae = mean_absolute_error(validation["count"], predictions)
        mlflow.log_metric("val_mae", mae)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(validation["count"], predictions, s=2, alpha=0.3)
        ax.plot([0, 1000], [0, 1000], color="grey")
        ax.set_xlabel("actual bikes per hour")
        ax.set_ylabel("predicted bikes per hour")
        ax.set_title(f"depth {depth}: MAE {mae:.1f}")
        mlflow.log_figure(fig, "actual_vs_predicted.png")
        plt.close(fig)

        mlflow.sklearn.log_model(model, artifact_path="model", serialization_format="cloudpickle")

        print(f"depth {depth}: validation MAE {mae:.1f} bikes/hour")
