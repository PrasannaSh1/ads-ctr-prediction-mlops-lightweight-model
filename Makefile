.PHONY: install pipeline test lint api docker-build

install:
	pip install -e ".[dev]"

pipeline:
	dvc repro

test:
	pytest -q

lint:
	ruff check src/ tests/

api:
	uvicorn taobao_ctr.serving.api:create_production_app --factory --host 0.0.0.0 --port 8000

docker-build:
	docker build -t taobao-ctr .
