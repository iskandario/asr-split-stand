"""Оркестрация прогонов стенда.

  uv run python runner/run_experiment.py configs/experiment.yaml          # весь план (продолжает с места остановки)
  uv run python runner/run_experiment.py configs/aa.yaml
  uv run python runner/run_experiment.py configs/smoke.yaml --dry-run    # только показать план

На каждый прогон: новый клиентский контейнер (чистый браузер, квота CPU из уровня «устройство»),
профиль сети внутри него, а после — манифест results/raw/<run_id>.json с метаданными.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
SERVER_HOST = os.environ.get("STAND_SERVER", "http://localhost:8000")
PROM = os.environ.get("STAND_PROMETHEUS", "http://localhost:9090")


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "include" in cfg:
        base = load_config(path.parent / cfg.pop("include"))
        base.update(cfg)
        cfg = base
    return cfg


def sh(*args: str, env: dict | None = None, check: bool = True) -> str:
    r = subprocess.run(args, capture_output=True, text=True, env={**os.environ, **(env or {})}, cwd=ROOT)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(args)}\n{r.stderr[-2000:]}")
    return r.stdout.strip()


def build_plan(cfg: dict) -> list[dict]:
    plan = []
    for block in cfg["blocks"]:
        f = block["factors"]
        for combo in itertools.product(f["strategy"], f["network"], f["device"]):
            for rep in range(block["repetitions"]):
                plan.append({"block": block["name"], "strategy": combo[0], "network": combo[1],
                             "device": combo[2], "rep": rep})
    if cfg.get("randomize_order"):
        random.Random(cfg["seed"]).shuffle(plan)
    for i, p in enumerate(plan):
        p["order"] = i
        p["run_id"] = f"{cfg['name']}-{i:03d}-{p['strategy']}-{p['network']}-{p['device']}-r{p['rep']}"
    return plan


def fragments(split: str, n: int) -> list[dict]:
    with (ROOT / "data" / "manifest.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    return [{"id": r["id"], "duration_s": float(r["duration_s"])} for r in rows[:n]]


def network_spec(cfg: dict, name: str) -> dict:
    p = dict(cfg["networks"][name])
    if "alternate" in p:
        p["alternate"] = [dict(cfg["networks"][x]) for x in p["alternate"]]
    return p


def run_spec(cfg: dict, p: dict) -> dict:
    return {
        "run_id": p["run_id"], "server_url": cfg["server_url"], "run_timeout_s": cfg["run_timeout_s"],
        "network": network_spec(cfg, p["network"]),
        "client": {**cfg["client"], "strategy": p["strategy"],
                   "fragments": fragments("test", cfg["fragments_per_run"]),
                   "warmup": fragments("tune", cfg["warmup_fragments"])},
    }


# ---------- метаданные ----------
def git_info() -> dict:
    try:
        return {"commit": sh("git", "rev-parse", "HEAD"), "dirty": bool(sh("git", "status", "--porcelain"))}
    except RuntimeError:
        return {"commit": None, "dirty": None}


def image_info() -> dict:
    out = {}
    for svc, img in (("server", "asr-stand-server:1.0"), ("client", "asr-stand-client:1.0"),
                     ("prometheus", "prom/prometheus:v2.55.1")):
        try:
            j = json.loads(sh("docker", "image", "inspect", img))[0]
            out[svc] = {"image": img, "id": j["Id"], "repo_digests": j.get("RepoDigests", []),
                        "created": j.get("Created")}
        except (RuntimeError, IndexError, json.JSONDecodeError):
            out[svc] = {"image": img, "id": None}
    return out


def host_info() -> dict:
    info = {"os": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
            "python": platform.python_version()}
    if sys.platform == "darwin":
        info["cpu"] = sh("sysctl", "-n", "machdep.cpu.brand_string", check=False)
        info["mem_bytes"] = int(sh("sysctl", "-n", "hw.memsize", check=False) or 0)
        info["on_ac_power"] = "AC Power" in sh("pmset", "-g", "batt", check=False)
    try:
        d = json.loads(sh("docker", "info", "--format", "{{json .}}"))
        info["docker"] = {"server_version": d.get("ServerVersion"), "vm_ncpu": d.get("NCPU"),
                          "vm_mem_bytes": d.get("MemTotal"), "kernel": d.get("KernelVersion"),
                          "os": d.get("OperatingSystem")}
    except (RuntimeError, json.JSONDecodeError):
        pass
    return info


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def server_metrics() -> dict:
    """Снимок счётчиков сервера: CPU процесса и число распознаваний."""
    try:
        text = urllib.request.urlopen(f"{SERVER_HOST}/metrics", timeout=5).read().decode()
    except OSError:
        return {}
    out = {}
    for key in ("process_cpu_seconds_total", "asr_inference_seconds_count", "asr_inference_seconds_sum"):
        m = re.search(rf"^{key} ([0-9.e+-]+)$", text, re.M)
        if m:
            out[key] = float(m.group(1))
    return out


def prom_peak_cpu(seconds: int) -> float | None:
    q = f"max_over_time(rate(process_cpu_seconds_total{{job=\"asr-server\"}}[6s])[{max(seconds, 10)}s:2s])"
    try:
        r = json.loads(urllib.request.urlopen(f"{PROM}/api/v1/query?query={urllib.parse.quote(q)}", timeout=5).read())
        return float(r["data"]["result"][0]["value"][1])
    except (OSError, KeyError, IndexError, ValueError):
        return None


def wait_server(timeout: float = 180) -> None:
    t = time.time()
    while time.time() - t < timeout:
        try:
            urllib.request.urlopen(f"{SERVER_HOST}/health", timeout=3)
            return
        except OSError:
            time.sleep(2)
    raise RuntimeError("сервер не поднялся")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="выполнить не больше N прогонов")
    a = ap.parse_args()

    cfg = load_config(a.config)
    plan = build_plan(cfg)
    RAW.mkdir(parents=True, exist_ok=True)
    todo = [p for p in plan if not (RAW / f"{p['run_id']}.json").exists()]
    print(f"{cfg['name']}: в плане {len(plan)} прогонов, осталось {len(todo)}")
    if a.dry_run:
        for p in todo[:20]:
            print(" ", p["run_id"])
        return 0

    sh("docker", "compose", "up", "-d", "--wait", "server", "prometheus")
    wait_server()
    static = {"git": git_info(), "images": image_info(), "host": host_info(),
              "config_file": str(a.config), "config_sha256": sha256_file(a.config),
              "config_resolved": cfg, "data_manifest_sha256": sha256_file(ROOT / "data" / "manifest.csv")}
    failures = 0
    for k, p in enumerate(todo[: a.limit] if a.limit else todo, 1):
        spec = run_spec(cfg, p)
        dev = cfg["devices"][p["device"]]
        started = datetime.now(timezone.utc)
        m0, t0 = server_metrics(), time.time()
        print(f"[{k}/{len(todo)}] {p['run_id']}", flush=True)
        r = subprocess.run(
            ["docker", "compose", "--profile", "run", "run", "--rm", "--no-deps",
             "--name", f"asr-client-{p['order']:03d}-{os.getpid()}", "client"],
            cwd=ROOT, capture_output=True, text=True,
            env={**os.environ, "RUN_SPEC": json.dumps(spec, ensure_ascii=False), "CLIENT_CPUS": str(dev["cpus"])},
        )
        dur = time.time() - t0
        m1 = server_metrics()
        ok = r.returncode == 0 and (RAW / f"{p['run_id']}.fragments.jsonl").exists()
        failures += 0 if ok else 1
        cpu = None
        if "process_cpu_seconds_total" in m0 and "process_cpu_seconds_total" in m1:
            cpu = (m1["process_cpu_seconds_total"] - m0["process_cpu_seconds_total"]) / dur
        manifest = {
            "run_id": p["run_id"], "experiment": cfg["name"], "block": p["block"], "order": p["order"],
            "params": {"strategy": p["strategy"], "network": p["network"], "device": p["device"]},
            "levels": {"network": spec["network"], "device": dev}, "repetition": p["rep"], "seed": cfg["seed"],
            "status": "ok" if ok else "failed", "exit_code": r.returncode,
            "started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_s": dur,
            "server": {"cpu_mean_cores": cpu, "cpu_peak_cores": prom_peak_cpu(int(dur)),
                       "inferences": (m1.get("asr_inference_seconds_count", 0) - m0.get("asr_inference_seconds_count", 0))},
            "stdout_tail": r.stdout[-1000:], "stderr_tail": r.stderr[-3000:] if not ok else "",
            **static,
        }
        name = p["run_id"] + (".json" if ok else ".failed.json")
        (RAW / name).write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        if not ok:
            print(f"   ОШИБКА (код {r.returncode}): {r.stderr[-500:]}", flush=True)
        if cfg.get("pause_s"):
            time.sleep(cfg["pause_s"])
    print(f"готово, ошибок: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
