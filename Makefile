.PHONY: setup synthetic-data pipeline-synthetic test lint api docker-build demo

VENV := .venv
PY := $(VENV)/bin/python

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

synthetic-data:
	$(PY) scripts/setup_synthetic_data.py --data-dir ./data --impressions 4000 --behaviour 20000 --seed 0

pipeline-synthetic: synthetic-data
	$(PY) -m dvc repro

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src/ tests/ scripts/

api:
	$(PY) -m uvicorn taobao_ctr.serving.api:create_production_app --factory --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -f docker/Dockerfile.api -t taobao-ctr-api .

demo: pipeline-synthetic
	docker compose up --build
