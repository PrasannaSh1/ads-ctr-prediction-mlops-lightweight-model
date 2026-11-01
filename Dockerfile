# One image for training and serving. The torch wheel from PyPI ships with CUDA, so the container
# uses the GPU when run with `--gpus all` (needs the NVIDIA Container Toolkit on the host).
FROM python:3.13-slim

WORKDIR /app

# libgomp1: OpenMP runtime needed by LightGBM
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir .

COPY params.yaml dvc.yaml ./

EXPOSE 8000

# Default: serve the API. To run a pipeline stage instead:
#   docker run --gpus all -v $(pwd):/app <image> python -m taobao_ctr.pipeline train_deep --name "3-tower co-train"
CMD ["uvicorn", "taobao_ctr.serving.api:create_production_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
