"""The FastAPI app: /health, /model-info and /score (user + candidate ad ids -> predicted CTR).

Run with:  uvicorn taobao_ctr.serving.api:create_production_app --factory --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, HTTPException

from taobao_ctr.serving.schemas import HealthResponse, ModelInfoResponse, ScoreRequest, ScoreResponse


def create_app(scorer, model_name="taobao-ctr-3tower", alias="champion"):
    """Build the app around an already created TwoStageScorer."""
    app = FastAPI(title="taobao-ctr-mlops serving API")

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(status="ok")

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info():
        return ModelInfoResponse(
            model_name=model_name,
            alias=alias,
            cascade_top_frac=scorer.cascade_top_frac,
            n_ads_cached=len(scorer.ad_cache["ad_ids"]),
        )

    @app.post("/score", response_model=ScoreResponse)
    def score(request: ScoreRequest):
        try:
            scores = scorer.score(request.user_id, request.ad_ids)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        scores_by_ad = {}
        for ad_id, value in zip(request.ad_ids, scores.tolist(), strict=True):
            scores_by_ad[ad_id] = value
        return ScoreResponse(user_id=request.user_id, scores=scores_by_ad)

    return app


def create_production_app():
    """Load the data and the champion model, build the caches, and return the app."""
    import json
    import os

    import mlflow.pytorch
    import torch

    from taobao_ctr.config import load_config
    from taobao_ctr.models.towers import ThreeTower
    from taobao_ctr.pipeline import load_dataset
    from taobao_ctr.serving.ad_cache import build_ad_cache
    from taobao_ctr.serving.scorer import TwoStageScorer, UserFeatureCache
    from taobao_ctr.training.trainer import default_hp

    config = load_config()
    dataset, train_rows, fit_rows, val_rows, test_rows = load_dataset(config)
    device = torch.device(config["device"])
    artifact_dir = config["paths"]["artifact_dir"]

    mlflow.set_tracking_uri(config["paths"]["mlflow_tracking_uri"])
    try:
        model = mlflow.pytorch.load_model("models:/taobao-ctr-3tower@champion", map_location=device)
    except Exception:
        # The registry is not reachable: load the champion weights saved by the register stage
        with open(os.path.join(artifact_dir, "registered.json")) as f:
            key = json.load(f)["source"]  # for example "3-tower co-train [default]"
        name, variant = key.rsplit(" [", 1)
        variant = variant.rstrip("]")
        weights_path = os.path.join(artifact_dir, "models", name, variant + ".pt")
        model = ThreeTower(default_hp(config), dataset.sizes, dataset.seq_len)
        model.load_state_dict(torch.load(weights_path, map_location=device))

    ad_cache = build_ad_cache(model, dataset, device)
    user_cache = UserFeatureCache(dataset)
    scorer = TwoStageScorer(model, ad_cache, user_cache, device, config["serve"]["cascade_top_frac"])
    return create_app(scorer)
