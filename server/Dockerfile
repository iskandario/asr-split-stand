# Сервер распознавания. Базовый образ и uv зафиксированы, модель скачивается по ревизии при сборке.
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv
WORKDIR /srv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
ARG MODEL_REPO=Systran/faster-whisper-base
ARG MODEL_REVISION=ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66
RUN uv run --frozen python -c "from huggingface_hub import snapshot_download as s; \
s('${MODEL_REPO}', revision='${MODEL_REVISION}', local_dir='/models/faster-whisper-base', \
allow_patterns=['config.json','model.bin','tokenizer.json','vocabulary.txt'])"
COPY app ./app
ENV MODEL_DIR=/models/faster-whisper-base CPU_THREADS=1 COMPUTE_TYPE=int8 HF_HUB_OFFLINE=1
EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=3s --retries=30 CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')"
CMD ["uv", "run", "--frozen", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
