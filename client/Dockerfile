# Клиент стенда: Chromium (Playwright) + страница с планировщиком + локальный Whisper в Wasm.
# Всё нужное для локального распознавания (transformers.js, wasm, веса) кладётся в образ при сборке.

FROM node:22.12.0-slim AS vendor
WORKDIR /v
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --ignore-scripts --no-audit --no-fund \
 && mkdir /out \
 && cp node_modules/@huggingface/transformers/dist/transformers.min.js \
       node_modules/@huggingface/transformers/dist/ort-wasm-simd-threaded.jsep.mjs \
       node_modules/@huggingface/transformers/dist/ort-wasm-simd-threaded.jsep.wasm /out/

FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev \
 && uv run --frozen playwright install --with-deps --only-shell chromium \
 && rm -rf /var/lib/apt/lists/*
ARG MODEL_REPO=onnx-community/whisper-base
ARG MODEL_REVISION=27a07b4b52b3a4c5886134bb4320e93e1a4bbc7c
RUN uv run --frozen python -c "from huggingface_hub import snapshot_download as s; \
s('${MODEL_REPO}', revision='${MODEL_REVISION}', local_dir='/models/whisper-base', allow_patterns=[ \
'config.json','generation_config.json','preprocessor_config.json','tokenizer.json','tokenizer_config.json', \
'special_tokens_map.json','added_tokens.json','vocab.json','normalizer.json', \
'onnx/encoder_model_quantized.onnx','onnx/decoder_model_merged_quantized.onnx'])"
COPY --from=vendor /out /app/vendor
COPY web /app/web
COPY driver /app/driver
ENV HF_HUB_OFFLINE=1
ENTRYPOINT ["uv", "run", "--frozen", "--no-sync", "python", "/app/driver/run_one.py"]
