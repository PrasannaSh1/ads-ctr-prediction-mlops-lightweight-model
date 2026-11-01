# Taobao Ad CTR Prediction (3-tower model + MLOps)

Predicts the probability that a user clicks a display ad (pCTR) on Alibaba's Taobao ads dataset
(26.6M impressions, 1.14M users, 723M-row behaviour log). Six models are trained and compared;
the 3-tower co-train model is registered in MLflow and served by a FastAPI two-stage scorer.

## Models

| Model | Test AUC |
| --- | --- |
| LightGBM | 0.6532 |
| Transformer (BST) | 0.6440 |
| 3-tower co-train | 0.6434 |
| MLP | 0.6420 |
| MLP + sequence | 0.6416 |
| Two-tower | 0.6388 |

The 3-tower model is deployed: its ad side can be cached, and its cheap two-tower head ranks all
candidates before the full model re-scores the top 40%.

## Project layout

```
params.yaml            all settings (paths, device, model, training, tuning)
dvc.yaml               the pipeline: data -> features -> training -> tuning -> evaluation -> registry
src/taobao_ctr/
  data/                load + encode impressions, behaviour-history store
  features/            time-ordered click-rate features, LightGBM features
  models/              MLP, MLP + sequence, Transformer, Two-tower, 3-tower
  training/            training loop, LightGBM, Optuna tuning
  evaluation/          results table, cascades and blends
  tracking/            MLflow model registry
  monitoring/          calibration, AUC per day, PSI drift report
  serving/             ad cache, two-stage scorer, FastAPI app
tests/                 unit tests
notebooks/             the original research notebook
```

## Run it (GPU machine)

1. Install (Python 3.13):
   ```bash
   pip install -e ".[dev]"
   ```
2. Set up Kaggle credentials (`~/.kaggle/kaggle.json`). The dataset
   (`zerotrustai/taobao-ad-display-click`, ~24 GB) downloads into `data/` on the first run.
3. Check the paths and `device: "cuda"` in `params.yaml`.
4. Run the whole pipeline:
   ```bash
   dvc repro
   dvc metrics show
   ```
5. Browse runs: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
6. Serve the model:
   ```bash
   make api                    # or: docker compose up --build
   curl -X POST http://localhost:8000/score -H "Content-Type: application/json" \
        -d '{"user_id": 1701, "ad_ids": [649, 47, 659]}'
   ```

## Pipeline stages

`prepare_impressions` -> `build_history` -> `click_rates` -> `train_lightgbm` + `train_deep` (5 models)
-> `tune` (Optuna, 5 models) -> `evaluate` -> `combine` -> `register` -> `monitor`

Each stage is `python -m taobao_ctr.pipeline <stage>`; DVC reruns only what changed.

## Tests and CI

`pytest` runs model shape tests and a label-leakage test for the click-rate features.
GitHub Actions runs ruff + pytest on every push, and pushes a Docker image to GHCR on version tags.
