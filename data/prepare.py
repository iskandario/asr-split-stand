"""Подготовка аудио для стенда.

Скачивает один parquet-файл тестовой части Golos (crowd, 10 ч) по зафиксированной
ревизии Hugging Face, отбирает фрагменты 2–10 с по сиду и пишет WAV 16 кГц моно.
Список файлов с SHA-256 и эталонными текстами — data/manifest.csv (лежит в git).

Режимы:
  python data/prepare.py          # скачать и собрать (если manifest.csv есть — сверить SHA-256)
  python data/prepare.py --check  # только сверить SHA-256 уже собранных файлов
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "data" / "audio"
MANIFEST = ROOT / "data" / "manifest.csv"

REPO = "bond005/sberdevices_golos_10h_crowd"
REVISION = "e634b6b810e4d30c81b4c6d8262379fe8b9f708c"
PARQUET = "data/test-00000-of-00003-cad0cfaddbc8fa71.parquet"
SEED = 20261116
N_TUNE, N_TEST = 20, 60
MIN_S, MAX_S = 2.0, 10.0
SR = 16000


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check() -> int:
    if not MANIFEST.exists():
        print("нет data/manifest.csv — запустите make data", file=sys.stderr)
        return 1
    bad = 0
    with MANIFEST.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = ROOT / row["file"]
            if not p.exists() or sha256(p) != row["sha256"]:
                print(f"НЕ СОВПАДАЕТ: {row['file']}", file=sys.stderr)
                bad += 1
    print("SHA-256: все файлы совпадают" if not bad else f"SHA-256: расхождений {bad}")
    return 1 if bad else 0


def build() -> int:
    import pandas as pd
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, PARQUET, repo_type="dataset", revision=REVISION)
    df = pd.read_parquet(path)
    rows = []
    for i, r in df.iterrows():
        audio = r["audio"]
        data, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SR:
            continue
        dur = len(data) / SR
        text = str(r["transcription"]).strip()
        if MIN_S <= dur <= MAX_S and len(text.split()) >= 2:
            rows.append((i, data, dur, text))
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(rows), size=N_TUNE + N_TEST, replace=False)
    AUDIO.mkdir(parents=True, exist_ok=True)
    out = []
    for k, j in enumerate(idx):
        i, data, dur, text = rows[j]
        split = "tune" if k < N_TUNE else "test"
        fid = f"{split}_{k if split == 'tune' else k - N_TUNE:03d}"
        f = AUDIO / f"{fid}.wav"
        sf.write(f, data, SR, subtype="PCM_16")
        out.append({"id": fid, "split": split, "file": f"data/audio/{fid}.wav",
                    "duration_s": f"{dur:.3f}", "sha256": sha256(f), "text": text,
                    "source_row": int(i)})
    if MANIFEST.exists():
        old = {r["id"]: r["sha256"] for r in csv.DictReader(MANIFEST.open(encoding="utf-8"))}
        diff = [r["id"] for r in out if old.get(r["id"]) != r["sha256"]]
        if diff:
            print(f"ВНИМАНИЕ: файлы отличаются от manifest.csv: {diff[:5]}…", file=sys.stderr)
            return 1
        print(f"Собрано {len(out)} фрагментов, SHA-256 совпадают с manifest.csv")
        return 0
    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    tot = sum(float(r["duration_s"]) for r in out)
    print(f"Собрано {len(out)} фрагментов ({tot:.0f} с), manifest.csv создан")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    sys.exit(check() if a.check else build())
