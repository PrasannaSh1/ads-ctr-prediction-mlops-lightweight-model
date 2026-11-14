"""MLflow helpers: point MLflow at the tracking store, and register the champion model.

The selected 3-tower model is registered as "taobao-ctr-3tower" with the alias "champion".
Serving then loads "models:/taobao-ctr-3tower@champion" without knowing which run made it.
"""
import json
import os

import mlflow
import mlflow.pytorch


def init_tracking(tracking_uri):
    mlflow.set_tracking_uri(tracking_uri)


def already_registered(registered_path, key):
    """True if registered.json says this exact model was registered before."""
    if not os.path.exists(registered_path):
        return False
    with open(registered_path) as f:
        info = json.load(f)
    return info.get("source") == key


def register_champion(model, key, hp, test_metrics, val_auc, registered_path,
                      model_name="taobao-ctr-3tower", alias="champion", fallback_weights_path=None):
    """Log the model to MLflow, register it as a new version, and point the alias at it.

    If the registry step fails, the raw weights file is logged as an artifact instead,
    so nothing is lost.
    """
    with mlflow.start_run(run_name="register " + model_name):
        params = {"source": key}
        params.update(hp)
        mlflow.log_params(params)
        mlflow.log_metrics({
            "test_AUC": test_metrics["AUC"],
            "val_AUC": val_auc,
            "test_LogLoss": test_metrics["LogLoss"],
            "test_PR_AUC": test_metrics["PR_AUC"],
        })
        try:
            try:
                # Recent MLflow. The pickle format is used because the default format needs
                # example inputs to trace the model.
                mlflow.pytorch.log_model(model, name="model", registered_model_name=model_name,
                                         serialization_format="pickle")
            except TypeError:
                # Older MLflow, which takes the name as a plain argument and has no pickle option
                mlflow.pytorch.log_model(model, "model", registered_model_name=model_name)
            client = mlflow.MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")
            newest_version = 0
            for v in versions:
                newest_version = max(newest_version, int(v.version))
            client.set_registered_model_alias(model_name, alias, newest_version)
            os.makedirs(os.path.dirname(registered_path), exist_ok=True)
            with open(registered_path, "w") as f:
                json.dump({"source": key, "version": newest_version}, f)
            print(f"registered {model_name} version {newest_version} as @{alias}")
        except Exception as e:
            print("model registry step failed:", e)
            if fallback_weights_path is not None:
                mlflow.log_artifact(str(fallback_weights_path))
