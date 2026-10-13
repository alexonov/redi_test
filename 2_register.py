"""Pick the run with the lowest validation error, register its model, label it "champion".

Registering = "we keep this one". The champion label = "this is the one the business uses".
"""

import mlflow
from mlflow import MlflowClient

import bikes


def main() -> None:
    model_name = bikes.setup_mlflow()

    best = mlflow.search_runs(order_by=["metrics.val_mae ASC"], max_results=1).iloc[0]
    print(f"best run: {best['tags.mlflow.runName']} ({best.run_id}), validation MAE {best['metrics.val_mae']:.1f}")

    version = mlflow.register_model(f"runs:/{best.run_id}/model", model_name)
    MlflowClient().set_registered_model_alias(model_name, "champion", version.version)
    print(f"registered {model_name} version {version.version} and labelled it 'champion'")


if __name__ == "__main__":
    main()
