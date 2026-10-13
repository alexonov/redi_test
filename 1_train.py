"""Train one model and log it to MLflow as one run.

Your job: fill in the three TODOs so every run is recorded in MLflow.
Stuck? The finished version is in solution/1_train.py.

Then: change the settings below, run the file again, compare runs on the Experiments page.
"""

import mlflow
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

import bikes

# ---- Settings: change these between runs ----
MODEL = "linear"        # "linear" or "forest"
FEATURES = "raw"        # "raw", "hour_onehot" or "mine"
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

    # Everything inside this "with" block belongs to one run in MLflow.
    with mlflow.start_run(run_name=f"{MODEL}-{FEATURES}") as run:
        params = {"model": MODEL, "features": FEATURES}
        if MODEL == "forest":
            params.update({"n_trees": N_TREES, "max_depth": MAX_DEPTH})

        # TODO 1: record the settings of this run.
        #         Hint: mlflow.log_params(...) takes a dictionary.

        model = bikes.build_model(MODEL, N_TREES, MAX_DEPTH)
        model.fit(x_train, y_train)

        predictions = model.predict(x_val)
        mae = mean_absolute_error(y_val, predictions)
        rmse = root_mean_squared_error(y_val, predictions)

        # TODO 2: record the two scores. Call them "val_mae" and "val_rmse"
        #         (script 2 looks for "val_mae").
        #         Hint: mlflow.log_metrics(...) also takes a dictionary.

        # TODO 3: save the trained model inside the run. Copy this line and remove the "#":
        # mlflow.sklearn.log_model(model, artifact_path="model", input_example=x_val.head(3), serialization_format="cloudpickle")

    print(f"run {run.info.run_id}: {MODEL} + {FEATURES} -> validation MAE {mae:.1f} bikes/hour, RMSE {rmse:.1f}")


if __name__ == "__main__":
    main()
