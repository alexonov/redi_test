"""Finished version of 1_train.py, with the three TODOs filled in.

Copy the lines you need into 1_train.py in the main folder. Don't run this file from here:
it needs bikes.py, which lives one folder up.
"""

import mlflow
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

import bikes

# ---- Settings: change these between runs ----
MODEL = "linear"        # "linear" or "forest"
FEATURES = "raw"        # "raw" or "hour_onehot"
N_TREES = 100           # forest only: how many trees
MAX_DEPTH = 10          # forest only: how deep each tree may grow
# ----------------------------------------------


def main() -> None:
    bikes.setup_mlflow()
    data = bikes.load_bikes()
    features = bikes.make_features(data, FEATURES)

    train = bikes.split(data, bikes.TRAIN_MONTHS)
    validation = bikes.split(data, bikes.VALIDATION_MONTHS)
    x_train, y_train = features.loc[train.index], train[bikes.TARGET]
    x_val, y_val = features.loc[validation.index], validation[bikes.TARGET]

    with mlflow.start_run(run_name=f"{MODEL}-{FEATURES}") as run:
        params = {"model": MODEL, "features": FEATURES}
        if MODEL == "forest":
            params.update({"n_trees": N_TREES, "max_depth": MAX_DEPTH})
        mlflow.log_params(params)

        model = bikes.build_model(MODEL, N_TREES, MAX_DEPTH)
        model.fit(x_train, y_train)

        predictions = model.predict(x_val)
        mae = mean_absolute_error(y_val, predictions)
        rmse = root_mean_squared_error(y_val, predictions)
        mlflow.log_metrics({"val_mae": mae, "val_rmse": rmse})

        mlflow.sklearn.log_model(model, artifact_path="model", input_example=x_val.head(3), serialization_format="cloudpickle")

    print(f"run {run.info.run_id}: {MODEL} + {FEATURES} -> validation MAE {mae:.1f} bikes/hour, RMSE {rmse:.1f}")


if __name__ == "__main__":
    main()
