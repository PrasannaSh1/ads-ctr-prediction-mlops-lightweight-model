"""Pydantic request/response models for the FastAPI app (/health, /score, /model-info)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    user_id: int = Field(..., description="Raw user id (the `userid` column of user_profile.csv)")
    ad_ids: list[int] = Field(..., min_length=1, description="Candidate adgroup_id values to score")


class ScoreResponse(BaseModel):
    user_id: int
    scores: dict[int, float] = Field(..., description="ad_id -> predicted CTR")


class ModelInfoResponse(BaseModel):
    model_name: str
    alias: str
    cascade_top_frac: float
    n_ads_cached: int


class HealthResponse(BaseModel):
    status: str
