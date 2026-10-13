"""A new month of data arrived (July 2012). Is the champion still the best we can do?

1. Score the current champion on July.
2. Retrain the same recipe on all data up to June (a "challenger") and score it on July.
3. Register the challenger. If it beats the champion on July, move the champion label to it.
"""

import mlflow
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error

import bikes


def main() -> None:
    model_name = bikes.setup_mlflow()
    client = MlflowClient()

    champion_version = client.get_model_version_by_alias(model_name, "champion")
    recipe = mlflow.get_run(champion_version.run_id).data.params
    champion = mlflow.sklearn.load_model(f"models:/{model_name}@champion")

    data = bikes.load_bikes()
    features = bikes.make_features(data, recipe["features"])
    new_month = bikes.split(data, [bikes.NEW_MONTH])
    up_to_june = bikes.split(data, list(bikes.TRAIN_MONTHS) + list(bikes.VALIDATION_MONTHS))

    champion_mae = mean_absolute_error(new_month[bikes.TARGET], champion.predict(features.loc[new_month.index]))

    with mlflow.start_run(run_name=f"retrain-{recipe['model']}-{recipe['features']}") as run:
        mlflow.log_params({**recipe, "trained_up_to_month": max(bikes.VALIDATION_MONTHS)})
        challenger = bikes.build_model(
            recipe["model"], int(recipe.get("n_trees", 100)), int(recipe.get("max_depth", 10))
        )
        challenger.fit(features.loc[up_to_june.index], up_to_june[bikes.TARGET])
        challenger_mae = mean_absolute_error(new_month[bikes.TARGET], challenger.predict(features.loc[new_month.index]))
        mlflow.log_metrics({"new_month_mae": challenger_mae, "champion_new_month_mae": champion_mae})
        mlflow.sklearn.log_model(challenger, artifact_path="model", input_example=features.loc[new_month.index].head(3), serialization_format="cloudpickle")

    print(f"July error, champion (version {champion_version.version}): {champion_mae:.1f} bikes/hour")
    print(f"July error, retrained challenger:            {challenger_mae:.1f} bikes/hour")

    version = mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)
    if challenger_mae < champion_mae:
        client.set_registered_model_alias(model_name, "champion", version.version)
        print(f"challenger wins: version {version.version} is now the champion")
    else:
        print(f"champion stays: version {version.version} is registered but not promoted")


if __name__ == "__main__":
    main()
