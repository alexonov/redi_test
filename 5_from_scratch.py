"""Write a full MLflow run yourself, without the helpers in bikes.py.

Goal: try a decision tree with different max_depth values and record each try as one run.
Use the MLflow cheat sheet in GUIDE.md. Stuck? The finished version is in solution/5_from_scratch.py.
"""

import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

EMAIL = "you@example.com"  # the email you log in to Databricks with
FEATURES = ["hour", "weekday", "temp", "humidity", "windspeed"]

# TODO 1: choose where the runs go.
#         mlflow.set_experiment(f"/Users/{EMAIL}/bike-scratch")

bikes = pd.read_csv("data/bike_sharing_hourly.csv")
train = bikes.loc[bikes["year"] == 0]                                   # 2011
validation = bikes.loc[(bikes["year"] == 1) & (bikes["month"] <= 6)]    # Jan-Jun 2012

for depth in [3, 6, 10, 15]:
    # TODO 2: start a run here, and put everything below inside it (indent it).

    # TODO 3: record the setting: max_depth

    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(train[FEATURES], train["count"])
    predictions = model.predict(validation[FEATURES])
    mae = mean_absolute_error(validation["count"], predictions)

    # TODO 4: record the score: val_mae

    # TODO 5: save the model

    print(f"depth {depth}: validation MAE {mae:.1f} bikes/hour")

# BONUS: draw actual vs predicted with matplotlib and save it in the run with mlflow.log_figure.
# BONUS: delete your logging lines, add mlflow.sklearn.autolog() at the top, run again.
#        What does MLflow record by itself?
