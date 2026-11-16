"""Замер WebGPU против Wasm в настоящем Chromium с GPU (запускать на машине с видеокартой, не в контейнере).

  uv run python bench/webgpu/run_bench.py            # результаты -> results/webgpu/<дата>.json
Страница отдаётся с заголовками COOP/COEP (нужны для многопоточного Wasm).
"""
import csv, functools, http.server, json, platform, subprocess, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODES = ["wasm-1t", "wasm-4t", "webgpu-fp32q4", "webgpu-fp16"]
REPS = 3


class H(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        path = path.split("?")[0]
        if path.startswith("/audio/"):
            return str(ROOT / "data" / "audio" / path[len("/audio/"):])
        return str(HERE / path.lstrip("/"))

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm"}

    def log_message(self, *a):
        pass


def main():
    rows = list(csv.DictReader(open(ROOT / "data" / "manifest.csv", encoding="utf-8")))
    frags = [r["id"] for r in rows if r["split"] == "test"][:16]
    warm = [r["id"] for r in rows if r["split"] == "tune"][:2]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8765), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    out = {"started_at": datetime.now(timezone.utc).isoformat(), "host": platform.platform(), "machine": platform.machine(),
           "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
           "power": subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.splitlines()[0:1],
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "fragments": frags, "warmup": warm, "reps": REPS, "runs": []}
    with sync_playwright() as p:
        b = p.chromium.launch(headless=False, args=["--enable-unsafe-webgpu", "--enable-features=WebGPU", "--window-size=500,300"])
        out["chromium"] = b.version
        for mode in MODES:
            pg = b.new_page()
            pg.goto("http://127.0.0.1:8765/index.html")
            pg.wait_for_function("document.title === 'ready'", timeout=60000)
            if "info" not in out:
                out["info"] = pg.evaluate("bench.info()")
                print("браузер:", json.dumps(out["info"], ensure_ascii=False))
            t = time.time()
            try:
                r = pg.evaluate("([m, w, f, n]) => bench.run(m, w, f, n)", [mode, warm, frags, REPS])
                med = sorted(x["infer_ms"] for x in r["results"])[len(r["results"]) // 2]
                print(f"{mode}: загрузка {r['loadMs']/1000:.1f} с, медиана {med:.0f} мс на фразу ({time.time()-t:.0f} с)")
                out["runs"].append(r)
            except Exception as e:  # noqa: BLE001
                print(f"{mode}: ошибка {str(e)[:300]}")
                out["runs"].append({"mode": mode, "error": str(e)[:2000]})
            pg.close()
        b.close()
    dst = ROOT / "results" / "webgpu"
    dst.mkdir(parents=True, exist_ok=True)
    f = dst / f"{datetime.now().strftime('%Y%m%d-%H%M')}.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("сохранено:", f)


if __name__ == "__main__":
    sys.exit(main())
