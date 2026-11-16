"""Один прогон стенда внутри клиентского контейнера.

Получает описание прогона в переменной RUN_SPEC (JSON от раннера), задаёт профиль сети
через tc netem, запускает Chromium со страницей клиента и сохраняет:
  /results/<run_id>.fragments.jsonl  — по строке на фрагмент
  /results/<run_id>.client.json      — метаданные клиента (браузер, квота CPU, сеть, загрузка модели)
  /results/<run_id>.console.log      — консоль страницы
"""
from __future__ import annotations

import functools
import http.server
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

RESULTS = Path("/results")
ROOTS = {"/vendor/": "/app/vendor", "/models/": "/models", "/audio/": "/data/audio"}
IFACE = os.environ.get("NET_IFACE", "eth0")


# ---------- статика для страницы (localhost не проходит через eth0 и не замедляется) ----------
class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm",
                      ".json": "application/json", ".onnx": "application/octet-stream", ".wav": "audio/wav"}

    def translate_path(self, path: str) -> str:
        path = path.split("?", 1)[0]
        for prefix, root in ROOTS.items():
            if path.startswith(prefix):
                return str(Path(root) / path[len(prefix):])
        return str(Path("/app/web") / path.lstrip("/"))

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a) -> None:  # тихо
        pass


def serve() -> http.server.ThreadingHTTPServer:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8080), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ---------- сеть ----------
def tc(*args: str) -> str:
    return subprocess.run(["tc", *args], check=True, capture_output=True, text=True).stdout


def netem_args(p: dict) -> list[str]:
    a = ["qdisc", "replace", "dev", IFACE, "root", "netem", "limit", "10000"]
    if p.get("delay_ms"):
        a += ["delay", f"{p['delay_ms']}ms"]
        if p.get("jitter_ms"):
            a += [f"{p['jitter_ms']}ms", "distribution", "normal"]
    if p.get("loss_pct"):
        a += ["loss", f"{p['loss_pct']}%"]
    if p.get("rate"):
        a += ["rate", p["rate"]]
    return a


def apply_profile(p: dict) -> None:
    if not any(p.get(k) for k in ("delay_ms", "loss_pct", "rate")):
        clear_net()
        return
    tc(*netem_args(p))


def clear_net() -> None:
    subprocess.run(["tc", "qdisc", "del", "dev", IFACE, "root"], capture_output=True)


class Alternator(threading.Thread):
    """Профиль N4: переключение между двумя профилями каждые period_s секунд."""

    def __init__(self, phases: list[dict], period_s: float):
        super().__init__(daemon=True)
        self.phases, self.period = phases, period_s
        self.stop = threading.Event()
        self.timeline: list[dict] = []

    def run(self) -> None:
        k = 0
        while not self.stop.is_set():
            ph = self.phases[k % len(self.phases)]
            apply_profile(ph)
            self.timeline.append({"t_epoch_ms": time.time() * 1000, "phase": ph.get("name", str(k))})
            k += 1
            self.stop.wait(self.period)


def cpu_quota() -> str:
    try:
        return Path("/sys/fs/cgroup/cpu.max").read_text().strip()
    except OSError:
        return "unknown"


def main() -> int:
    spec = json.loads(os.environ["RUN_SPEC"])
    run_id = spec["run_id"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    console = (RESULTS / f"{run_id}.console.log").open("w", encoding="utf-8")
    net = spec["network"]
    alt = None
    t_start = time.time()
    srv = serve()
    try:
        # сеть задаётся до запуска браузера: прогрев идёт уже в условиях прогона
        if net.get("alternate"):
            alt = Alternator(net["alternate"], net["period_s"])
            alt.start()
        else:
            apply_profile(net)
        qdisc = tc("qdisc", "show", "dev", IFACE).strip()

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=["--disable-background-timer-throttling",
                                                               "--disable-renderer-backgrounding"])
            page = browser.new_page()
            page.on("console", lambda m: console.write(f"{time.time():.3f} {m.type} {m.text}\n"))
            page.on("pageerror", lambda e: console.write(f"{time.time():.3f} pageerror {e}\n"))
            page.goto(f"http://127.0.0.1:8080/index.html?server={spec['server_url']}")
            page.wait_for_function("window.stand && window.stand.status === 'idle'", timeout=30_000)
            page.evaluate("cfg => { window.stand.run(cfg); }", spec["client"])
            page.wait_for_function("window.stand.status !== 'running'", timeout=spec["run_timeout_s"] * 1000,
                                   polling=500)
            status = page.evaluate("window.stand.status")
            result = page.evaluate("window.stand.result")
            version = browser.version
            browser.close()
    finally:
        if alt:
            alt.stop.set()
            alt.join(timeout=2)
        clear_net()
        srv.shutdown()
        console.close()

    if status != "done":
        print(f"прогон {run_id} завершился с ошибкой: {result.get('error')}", file=sys.stderr)
        (RESULTS / f"{run_id}.client.json").write_text(json.dumps({"status": status, **result}, ensure_ascii=False))
        return 2
    with (RESULTS / f"{run_id}.fragments.jsonl").open("w", encoding="utf-8") as f:
        for rec in result["records"]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    meta = {
        "status": status, "run_id": run_id, "chromium": version, "cpu_quota": cpu_quota(),
        "nproc": os.cpu_count(), "qdisc_at_start": qdisc, "net_timeline": alt.timeline if alt else [],
        "driver_wall_s": time.time() - t_start, **result["meta"],
    }
    (RESULTS / f"{run_id}.client.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"{run_id}: {len(result['records'])} фрагментов, {meta['driver_wall_s']:.0f} с")
    return 0


if __name__ == "__main__":
    sys.exit(main())
