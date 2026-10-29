# taobao-ctr-mlops

Click-through-rate (pCTR) prediction on Alibaba's Taobao display-ads dataset, ported from a
tested research notebook into a production-style MLOps project: LightGBM baseline plus five deep
models, including a 3-tower engagement co-train model, served behind a two-stage FastAPI scorer.

**Status: work in progress, built in phases.** This README is a placeholder until Phase 8, when
the full write-up (results, architecture, how to run locally / Kaggle / Docker) lands. See
`docs/mlops_guide.md` (Phase 8) for a beginner-friendly tour of every MLOps piece once it exists.

## Phase 1 (current): project scaffold

- `pyproject.toml` — the `taobao_ctr` package, pinned dependencies, ruff + pytest config.
- `params.yaml` / `params.synthetic.yaml` — every setting from the notebook's `CFG` dict, grouped,
  with a tiny "synthetic" profile for local runs and CI.
- `src/taobao_ctr/config.py` — loads and validates params.yaml (pydantic), detects the platform
  (Kaggle / Colab / local), and resolves data/artifact paths from the environment.
- `src/taobao_ctr/**` — package skeleton; each module has a docstring saying what it will hold and
  which phase fills it in.

Nothing here runs an actual pipeline stage yet — that starts in Phase 2.
