"""MLflow tracking URI setup and idempotent registration of the selected 3-tower model as
`taobao-ctr-3tower` with alias `champion` (notebook sections 3 and 13). A serving job then
loads "the current champion" without knowing which run produced it:
`mlflow.pytorch.load_model("models:/taobao-ctr-3tower@champion")`.

Run-level logging itself (params/metrics/validation curve for one training run) is just plain
`mlflow.start_run()`/`log_params`/`log_metrics` calls made directly in pipeline.py where the
run happens -- there is no value in wrapping a single `with` block in another function.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.pytorch


def init_tracking(tracking_uri: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)


def already_registered(registered_path: str | Path, key: str) -> bool:
    path = Path(registered_path)
    return path.exists() and json.loads(path.read_text()).get("source") == key


def register_champion(
    model,
    key: str,
    hp: dict,
    test_metrics: dict,
    val_auc: float,
    registered_path: str | Path,
    model_name: str = "taobao-ctr-3tower",
    alias: str = "champion",
    fallback_weights_path: str | Path | None = None,
) -> None:
    """Logs the model, registers a new version, and points `alias` at it. If the registry step
    fails (e.g. no tracking server configured for model registration), the run still logs the
    raw weights file as an artifact so nothing is lost -- same fallback as the notebook."""
    with mlflow.start_run(run_name=f"register {model_name}"):
        mlflow.log_params({"source": key, **hp})
        mlflow.log_metrics({
            "test_AUC": test_metrics["AUC"], "val_AUC": val_auc,
            "test_LogLoss": test_metrics["LogLoss"], "test_PR_AUC": test_metrics["PR_AUC"],
        })
        try:
            try:  # recent MLflow: pickle format (its default traced format needs example inputs)
                mlflow.pytorch.log_model(model, name="model", registered_model_name=model_name, serialization_format="pickle")
            except TypeError:  # older MLflow
                mlflow.pytorch.log_model(model, "model", registered_model_name=model_name)
            client = mlflow.MlflowClient()
            version = max(int(v.version) for v in client.search_model_versions(f"name='{model_name}'"))
            client.set_registered_model_alias(model_name, alias, version)
            Path(registered_path).parent.mkdir(parents=True, exist_ok=True)
            Path(registered_path).write_text(json.dumps({"source": key, "version": version}))
            print(f"registered {model_name} version {version} as @{alias}")
        except Exception as e:  # the weights are still saved on disk either way
            print("model registry step failed:", e)
            if fallback_weights_path is not None:
                mlflow.log_artifact(str(fallback_weights_path))
