"""Сервер распознавания речи для стенда.

POST /transcribe  WAV 16 кГц моно -> {"text", "infer_ms", "queue_ms", "audio_s"}
GET  /ping        дешёвый зонд задержки
POST /probe       зонд пропускной способности (тело читается и отбрасывается)
GET  /health      модель загружена
GET  /metrics     метрики Prometheus
"""
from __future__ import annotations

import io
import os
import threading
import time
import wave

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/faster-whisper-base")
CPU_THREADS = int(os.environ.get("CPU_THREADS", "1"))
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "int8")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))

INFER = Histogram("asr_inference_seconds", "Время работы модели на фрагмент",
                  buckets=(.1, .2, .3, .5, .75, 1, 1.5, 2, 3, 5, 8, 13))
QUEUE = Histogram("asr_queue_seconds", "Ожидание модели в очереди сервера",
                  buckets=(0, .05, .1, .25, .5, 1, 2, 4, 8))

app = FastAPI(title="ASR server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
Instrumentator().instrument(app).expose(app)

model: WhisperModel | None = None
lock = threading.Lock()  # одна модель — одно распознавание за раз, остальные ждут в очереди


@app.on_event("startup")
def load() -> None:
    global model
    model = WhisperModel(MODEL_DIR, device="cpu", compute_type=COMPUTE_TYPE, cpu_threads=CPU_THREADS)


def decode_wav(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data)) as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1 or w.getframerate() != 16000:
            raise ValueError("нужен WAV PCM16 моно 16 кГц")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def transcribe(audio: np.ndarray, t_arrived: float) -> dict:
    with lock:
        t_start = time.perf_counter()
        segments, _ = model.transcribe(
            audio, language="ru", task="transcribe", beam_size=1, best_of=1,
            temperature=0.0, condition_on_previous_text=False, without_timestamps=True,
            vad_filter=False, max_new_tokens=MAX_NEW_TOKENS,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        t_end = time.perf_counter()
    QUEUE.observe(t_start - t_arrived)
    INFER.observe(t_end - t_start)
    return {"text": text, "infer_ms": (t_end - t_start) * 1000, "queue_ms": (t_start - t_arrived) * 1000,
            "audio_s": len(audio) / 16000}


@app.post("/transcribe")
async def transcribe_endpoint(request: Request) -> dict:
    body = await request.body()
    t_arrived = time.perf_counter()  # отсчёт после приёма тела: время передачи считает клиент
    try:
        audio = decode_wav(body)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, str(e)) from e
    return await run_in_threadpool(transcribe, audio, t_arrived)


@app.get("/ping")
def ping() -> dict:
    return {"ok": True}


@app.post("/probe")
async def probe(request: Request) -> dict:
    return {"n": len(await request.body())}


@app.get("/health")
def health() -> dict:
    if model is None:
        raise HTTPException(503, "model not loaded")
    return {"ok": True, "model": MODEL_DIR, "cpu_threads": CPU_THREADS, "compute_type": COMPUTE_TYPE}
