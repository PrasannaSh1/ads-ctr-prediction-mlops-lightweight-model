"""FastAPI app: /health, /score (user + candidate ad ids -> pCTR), /model-info.

Run with:  uvicorn taobao_ctr.serving.api:create_production_app --factory --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from taobao_ctr.serving.schemas import HealthResponse, ModelInfoResponse, ScoreRequest, ScoreResponse
from taobao_ctr.serving.scorer import TwoStageScorer


def create_app(scorer: TwoStageScorer, model_name: str = "taobao-ctr-3tower", alias: str = "champion") -> FastAPI:
    app = FastAPI(title="taobao-ctr-mlops serving API")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        return ModelInfoResponse(
            model_name=model_name, alias=alias, cascade_top_frac=scorer.cascade_top_frac,
            n_ads_cached=len(scorer.ad_cache["ad_ids"]),
        )

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        try:
            scores = scorer.score(req.user_id, req.ad_ids)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return ScoreResponse(user_id=req.user_id, scores=dict(zip(req.ad_ids, scores.tolist(), strict=True)))

    return app


def create_production_app() -> FastAPI:
    import json

    import mlflow.pytorch
    import torch

    from taobao_ctr.config import load_config
    from taobao_ctr.models.towers import ThreeTower
    from taobao_ctr.pipeline import load_dataset
    from taobao_ctr.serving.ad_cache import build_ad_cache
    from taobao_ctr.serving.scorer import UserFeatureCache
    from taobao_ctr.training.trainer import default_hp

    cfg = load_config()
    dataset, *_ = load_dataset(cfg)
    device = torch.device(cfg.device)

    mlflow.set_tracking_uri(cfg.paths.mlflow_tracking_uri)
    try:
        model = mlflow.pytorch.load_model("models:/taobao-ctr-3tower@champion", map_location=device)
    except Exception:  # registry not reachable: load the champion weights saved by the register stage
        key = json.loads((cfg.paths.artifact_dir / "registered.json").read_text())["source"]
        name, variant = key.rsplit(" [", 1)
        weights_path = cfg.paths.artifact_dir / "models" / name / f"{variant.rstrip(']')}.pt"
        model = ThreeTower(default_hp(cfg), dataset.sizes, dataset.seq_len)
        model.load_state_dict(torch.load(weights_path, map_location=device))

    ad_cache = build_ad_cache(model, dataset, device)
    scorer = TwoStageScorer(model, ad_cache, UserFeatureCache(dataset), device, cfg.serve.cascade_top_frac)
    return create_app(scorer)
