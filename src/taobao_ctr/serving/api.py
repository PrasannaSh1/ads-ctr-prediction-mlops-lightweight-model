"""FastAPI app: /health, /score (user + candidate ad ids -> pCTR), /model-info.

`create_app(scorer, ...)` is the pure, test-friendly factory (dependency injection: pass in an
already-built TwoStageScorer). `create_production_app()` does the real wiring -- load params,
load the champion 3-tower from MLflow (or a fresh model if the registry step never ran), build
the ad/user caches -- and is meant to be run via uvicorn's `--factory` flag so none of that
heavy work happens at import time:

    uvicorn taobao_ctr.serving.api:create_production_app --factory --host 0.0.0.0 --port 8000
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
    import mlflow.pytorch

    from taobao_ctr.config import load_config
    from taobao_ctr.models.towers import ThreeTower
    from taobao_ctr.pipeline import load_dataset
    from taobao_ctr.serving.ad_cache import build_ad_cache
    from taobao_ctr.serving.scorer import UserFeatureCache
    from taobao_ctr.training.trainer import default_hp, get_device

    cfg = load_config()
    dataset, *_ = load_dataset(cfg)
    device = get_device()

    mlflow.set_tracking_uri(cfg.paths.mlflow_tracking_uri)
    try:
        model = mlflow.pytorch.load_model("models:/taobao-ctr-3tower@champion", map_location=device)
    except Exception:  # registry step never ran (e.g. first local demo) -- serve an untrained model
        model = ThreeTower(default_hp(cfg.params), dataset.sizes, dataset.seq_len)

    ad_cache = build_ad_cache(model, dataset, device)
    scorer = TwoStageScorer(model, ad_cache, UserFeatureCache(dataset), device, cfg.params.serve.cascade_top_frac)
    return create_app(scorer)
