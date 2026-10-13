"""Use the model the way the business would: ask for "the champion" by name, not by file.

Whoever runs this does not need to know which run, settings or version won.
"""

import mlflow
from mlflow import MlflowClient

import bikes

MONTH = 18  # June 2012, from the validation period


def main() -> None:
    model_name = bikes.setup_mlflow()
    champion = mlflow.sklearn.load_model(f"models:/{model_name}@champion")

    # The champion needs the same kind of inputs it was trained on: look up its feature set.
    champion_version = MlflowClient().get_model_version_by_alias(model_name, "champion")
    feature_set = mlflow.get_run(champion_version.run_id).data.params["features"]
    print(f"champion = {model_name} version {champion_version.version}, features '{feature_set}'")

    data = bikes.load_bikes()
    features = bikes.make_features(data, feature_set)
    june_working_days = bikes.split(data, [MONTH]).query("workingday.astype('str') == 'True'")
    june_working_days["predicted"] = champion.predict(features.loc[june_working_days.index])

    by_hour = june_working_days.groupby("hour")[[bikes.TARGET, "predicted"]].mean().round()
    print("average bikes per hour on a June working day: actual vs champion's forecast")
    print(by_hour.to_string())


if __name__ == "__main__":
    main()
