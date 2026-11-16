"""Анализ результатов стенда: сырые данные -> таблицы, статистика, графики.

  uv run python analysis/analyze.py                       # results/raw -> results/processed
  uv run python analysis/analyze.py --raw results/reference --out results/check   # пересчёт эталонных данных

Единица анализа — прогон: из каждого прогона берётся p95 задержки (и p50, WER, доля распознанных).
Сравнения — непараметрические: медиана по повторам, bootstrap-интервалы, Манн–Уитни, δ Клиффа, Холм.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import jiwer
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from num2words import num2words  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261116
CENSOR_MS = 30_000          # неудачный фрагмент считается задержкой 30 с (цензурирование)
RECOGNIZED_MS = 10_000      # «распознан» = текст пришёл не позже 10 с
EFFECT = -0.20              # практически значимый эффект из ПР3: −20 % p95
STRATS = ["S1", "S2", "S3", "S4"]
SIMPLE = ["S1", "S2", "S3"]
NETS = ["N1", "N2", "N3", "N4"]
DEVS = ["D1", "D2", "D3"]
COLORS = {"S1": "#4c72b0", "S2": "#dd8452", "S3": "#8c8c8c", "S4": "#2a9d55"}
NAMES = {"S1": "S1 локально", "S2": "S2 сервер", "S3": "S3 порог", "S4": "S4 адаптивная"}


# ---------- нормализация текста для WER ----------
def normalize(text: str) -> str:
    t = (text or "").lower().replace("ё", "е")
    t = re.sub(r"\d+", lambda m: " " + num2words(int(m.group()), lang="ru") + " ", t)
    t = re.sub(r"[^\w\s]|_", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def wer(refs: list[str], hyps: list[str]) -> float:
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    if not pairs:
        return float("nan")
    return jiwer.wer([r for r, _ in pairs], [h if h else "<пусто>" for _, h in pairs])


# ---------- статистика ----------
def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sign(np.subtract.outer(a, b)).mean())


def cliffs_label(d: float) -> str:
    d = abs(d)
    return "пренебрежимый" if d < 0.147 else "малый" if d < 0.33 else "средний" if d < 0.474 else "большой"


def boot_ci_median(x, seed=SEED) -> tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) < 3 or np.all(x == x[0]):
        return float(np.median(x)), float(np.median(x))
    r = stats.bootstrap((x,), np.median, confidence_level=0.95, n_resamples=5000,
                        random_state=np.random.default_rng(seed), method="percentile")
    return float(r.confidence_interval.low), float(r.confidence_interval.high)


def boot_ci_rel_diff(a, b, seed=SEED, n=5000) -> tuple[float, float, float]:
    """(median(a) − median(b)) / median(b) и 95 % percentile-интервал по независимому bootstrap."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    point = (np.median(a) - np.median(b)) / np.median(b)
    ma = np.median(rng.choice(a, (n, len(a))), axis=1)
    mb = np.median(rng.choice(b, (n, len(b))), axis=1)
    d = (ma - mb) / mb
    return float(point), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * pvals[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


# ---------- загрузка ----------
def load(raw: Path, experiment: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    refs = pd.read_csv(ROOT / "data" / "manifest.csv").set_index("id")["text"].to_dict()
    runs, frags = [], []
    for mf in sorted(raw.glob(f"{experiment}-*.json")):
        if mf.name.endswith(".failed.json") or mf.name.endswith(".client.json"):
            continue
        m = json.loads(mf.read_text())
        rid = m["run_id"]
        fj, cj = raw / f"{rid}.fragments.jsonl", raw / f"{rid}.client.json"
        if m["status"] != "ok" or not fj.exists():
            continue
        client = json.loads(cj.read_text())
        recs = [json.loads(line) for line in fj.read_text(encoding="utf-8").splitlines() if line]
        lat = np.array([CENSOR_MS if r["failed"] or r["latency_ms"] is None else r["latency_ms"] for r in recs])
        hyps = [normalize(r.get("text", "")) if not r["failed"] else "" for r in recs]
        rr = [normalize(refs[r["id"]]) for r in recs]
        lag = np.array([r["ready_ms"] - r["sched_ms"] for r in recs])
        runs.append({
            "run_id": rid, "block": m["block"], "order": m["order"], "rep": m["repetition"], **m["params"],
            "p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
            "mean": float(lat.mean()), "wer": wer(rr, hyps),
            "recognized": float(np.mean([(not r["failed"]) and r["latency_ms"] <= RECOGNIZED_MS for r in recs])),
            "failed": int(sum(r["failed"] for r in recs)),
            "server_share": float(np.mean([r.get("done_where") == "server" for r in recs])),
            "fallbacks": int(sum(r.get("fallback", False) for r in recs)),
            "traffic_kb": (client["traffic"]["audio_bytes"] + client["traffic"]["probe_bytes"]) / 1024,
            "lag_p95_ms": float(np.percentile(lag, 95)), "lag_max_ms": float(lag.max()),
            "local_load_ms": client.get("localLoadMs"),
            "server_cpu_mean": (m.get("server") or {}).get("cpu_mean_cores"),
            "server_cpu_peak": (m.get("server") or {}).get("cpu_peak_cores"),
            "started_at": m["started_at"], "chromium": client.get("chromium"),
            "transformers": client.get("transformersVersion"), "git_commit": m["git"]["commit"],
        })
        for r, h, ref, la in zip(recs, hyps, rr, lat):
            frags.append({"run_id": rid, **m["params"], "id": r["id"], "latency_ms": la, "failed": r["failed"],
                          "where": r.get("done_where"), "decided": r["where"], "fallback": r.get("fallback", False),
                          "ready_ms": r["ready_ms"], "est_mbps": r.get("est_mbps"), "hyp": h, "ref": ref})
    return pd.DataFrame(runs), pd.DataFrame(frags)


# ---------- таблицы ----------
def summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, n, d), g in df.groupby(["strategy", "network", "device"]):
        lo, hi = boot_ci_median(g["p95"])
        rows.append({"strategy": s, "network": n, "device": d, "n": len(g),
                     "p95_median": g["p95"].median(), "p95_ci_low": lo, "p95_ci_high": hi,
                     "p95_iqr": g["p95"].quantile(.75) - g["p95"].quantile(.25),
                     "p50_median": g["p50"].median(), "wer_median": g["wer"].median(),
                     "recognized_median": g["recognized"].median(), "server_share": g["server_share"].mean(),
                     "traffic_kb": g["traffic_kb"].median()})
    return pd.DataFrame(rows)


def compare_s4(df: pd.DataFrame) -> pd.DataFrame:
    """S4 против лучшей простой стратегии в каждом условии «сеть × устройство»."""
    rows = []
    for (n, d), g in df.groupby(["network", "device"]):
        med = g.groupby("strategy")["p95"].median()
        if "S4" not in med or not set(SIMPLE) & set(med.index):
            continue
        best = med[[s for s in SIMPLE if s in med]].idxmin()
        a, b = g[g.strategy == "S4"]["p95"].values, g[g.strategy == best]["p95"].values
        # H1 направленная (S4 быстрее), направление задано до эксперимента (ПР3) -> односторонний тест
        _, p = stats.mannwhitneyu(a, b, alternative="less")
        _, p2 = stats.mannwhitneyu(a, b, alternative="two-sided")
        rel, lo, hi = boot_ci_rel_diff(a, b)
        wa, wb = g[g.strategy == "S4"]["wer"].median(), g[g.strategy == best]["wer"].median()
        worst = med[[s for s in SIMPLE if s in med]].idxmax()
        rows.append({"network": n, "device": d, "best_simple": best, "p95_S4": np.median(a), "p95_best": np.median(b),
                     "rel_diff": rel, "ci_low": lo, "ci_high": hi, "p": p, "p_two_sided": p2, "cliff": cliffs_delta(a, b),
                     "wer_S4": wa, "wer_best": wb, "wer_diff_pp": (wa - wb) * 100,
                     "worst_simple": worst, "rel_vs_worst": (np.median(a) - med[worst]) / med[worst]})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"].tolist())
        out["effect"] = out["cliff"].map(cliffs_label)
        out["h1"] = (out["ci_high"] < EFFECT) & (out["p_holm"] < 0.05) & (out["wer_diff_pp"] <= 1)
        out["s4_not_worse"] = out["ci_high"] < 0.10
    return out


def regret(df: pd.DataFrame) -> pd.DataFrame:
    """Насколько каждая стратегия хуже лучшей в каждом условии (по медиане p95)."""
    med = df.groupby(["network", "device", "strategy"])["p95"].median().unstack()
    rel = med.div(med.min(axis=1), axis=0) - 1
    return rel


def aa_test(aa: pd.DataFrame, seed: int = SEED) -> dict:
    x = aa.sort_values("order")["p95"].values
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    ga, gb = x[idx[: len(x) // 2]], x[idx[len(x) // 2:]]
    _, p = stats.mannwhitneyu(ga, gb, alternative="two-sided")
    rho, p_trend = stats.spearmanr(np.arange(len(x)), x)
    cv = float(np.std(x, ddof=1) / np.mean(x))
    n = len(x) // 2
    mde = 2.8 * cv * np.sqrt(2 / n)  # минимальный обнаружимый эффект (α=0,05, мощность 80 %, норм. приближение)
    return {"n": len(x), "median_A": float(np.median(ga)), "median_B": float(np.median(gb)),
            "rel_diff": float((np.median(ga) - np.median(gb)) / np.median(gb)), "p": float(p),
            "cliff": cliffs_delta(ga, gb), "cv": cv, "mde_rel": float(mde), "spearman_rho": float(rho),
            "spearman_p": float(p_trend), "p95_values": x.tolist(), "groups": ["A" if i < n else "B" for i in np.argsort(idx)]}


# ---------- графики ----------
def plot_boxes(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(len(DEVS), len(NETS), figsize=(14, 10), sharey="row")
    rng = np.random.default_rng(0)
    for i, d in enumerate(DEVS):
        for j, n in enumerate(NETS):
            ax = axes[i, j]
            g = df[(df.network == n) & (df.device == d)]
            data = [g[g.strategy == s]["p95"].values / 1000 for s in STRATS]
            ax.boxplot(data, widths=.55, showfliers=False, medianprops={"color": "black"})
            for k, (s, v) in enumerate(zip(STRATS, data), 1):
                ax.scatter(k + rng.uniform(-.12, .12, len(v)), v, s=14, color=COLORS[s], zorder=3)
            ax.set_xticks(range(1, 5), STRATS)
            ax.set_title(f"{n} × {d}", fontsize=10)
            ax.grid(axis="y", alpha=.3)
            if j == 0:
                ax.set_ylabel("p95 задержки за прогон, с")
    fig.suptitle("p95 задержки по прогонам: стратегия × сеть × устройство (точки — отдельные прогоны)")
    fig.tight_layout()
    fig.savefig(out / "fig_p95_boxplots.png", dpi=140)
    plt.close(fig)


def plot_ecdf(fr: pd.DataFrame, out: Path) -> None:
    conds = [("N1", "D1"), ("N3", "D1"), ("N4", "D2"), ("N2", "D3")]
    fig, axes = plt.subplots(1, len(conds), figsize=(16, 4), sharey=True)
    for ax, (n, d) in zip(axes, conds):
        g = fr[(fr.network == n) & (fr.device == d)]
        for s in STRATS:
            v = np.sort(g[g.strategy == s]["latency_ms"].values / 1000)
            if len(v):
                ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=COLORS[s], label=NAMES[s])
        ax.set_xscale("log")
        ax.set_title(f"{n} × {d}")
        ax.set_xlabel("задержка фрагмента, с (лог.)")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("доля фрагментов")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("ECDF задержки отдельных фрагментов (все повторы вместе)")
    fig.tight_layout()
    fig.savefig(out / "fig_ecdf.png", dpi=140)
    plt.close(fig)


def plot_stability(df: pd.DataFrame, out: Path) -> tuple[float, float]:
    med = df.groupby(["strategy", "network", "device"])["p95"].transform("median")
    rel = df["p95"] / med
    rho, p = stats.spearmanr(df["order"], rel)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.scatter(df["order"], rel, s=10, c=[COLORS[s] for s in df.strategy])
    ax.axhline(1, color="black", lw=.8)
    ax.set_xlabel("порядковый номер прогона")
    ax.set_ylabel("p95 / медиана своей конфигурации")
    ax.set_title(f"Стабильность во времени: ρ Спирмена = {rho:+.3f}, p = {p:.3f}")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "fig_stability.png", dpi=140)
    plt.close(fig)
    return float(rho), float(p)


def plot_heatmap(reg: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    v = reg[STRATS].values * 100
    im = ax.imshow(np.minimum(v, 300), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(4), STRATS)
    ax.set_yticks(range(len(reg)), [f"{n} × {d}" for n, d in reg.index])
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.text(j, i, f"{v[i, j]:.0f}", ha="center", va="center", fontsize=8)
    ax.set_title("На сколько % p95 хуже лучшей стратегии условия")
    fig.colorbar(im, ax=ax, shrink=.7)
    fig.tight_layout()
    fig.savefig(out / "fig_regret_heatmap.png", dpi=140)
    plt.close(fig)


def plot_aa(aa: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.array(aa["p95_values"]) / 1000
    cols = ["#4c72b0" if g == "A" else "#dd8452" for g in aa["groups"]]
    ax.scatter(range(1, len(x) + 1), x, c=cols, s=40)
    ax.set_xlabel("номер прогона (A — синие, B — оранжевые)")
    ax.set_ylabel("p95, с")
    ax.set_title(f"A/A: S2 × N1 × D1, разница медиан {aa['rel_diff']*100:+.1f} %, p = {aa['p']:.2f}")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "fig_aa.png", dpi=140)
    plt.close(fig)


def plot_timeline(fr: pd.DataFrame, out: Path) -> None:
    g = fr[(fr.strategy == "S4") & (fr.network == "N4") & (fr.device == "D2")]
    if g.empty:
        return
    rid = sorted(g.run_id.unique())[0]
    g = g[g.run_id == rid].sort_values("ready_ms")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for where, c in (("local", COLORS["S1"]), ("server", COLORS["S2"])):
        h = g[g["where"] == where]
        ax.scatter(h.ready_ms / 1000, h.latency_ms / 1000, color=c, label=f"выполнено: {where}", s=30)
    fb = g[g["fallback"].astype(bool)]
    ax.scatter(fb.ready_ms / 1000, fb.latency_ms / 1000, facecolors="none", edgecolors="red", s=90, label="откат на локальное")
    ax2 = ax.twinx()
    ax2.step(g.ready_ms / 1000, g.est_mbps, where="post", color="gray", alpha=.6)
    ax2.set_ylabel("оценка канала, Мбит/с", color="gray")
    ax.set_xlabel("время от начала прогона, с")
    ax.set_ylabel("задержка фрагмента, с")
    ax.set_title(f"S4 при нестабильной сети N4 (прогон {rid})")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_s4_timeline.png", dpi=140)
    plt.close(fig)


# ---------- отчёт ----------
def fmt_s(ms: float) -> str:
    return f"{ms/1000:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "results" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "processed")
    ap.add_argument("--experiment", default="main")
    ap.add_argument("--aa", default="aa")
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)

    df, fr = load(a.raw, a.experiment)
    if df.empty:
        raise SystemExit(f"нет прогонов эксперимента {a.experiment} в {a.raw}")
    df.to_csv(out / "runs.csv", index=False)
    fr.to_csv(out / "fragments.csv", index=False)
    main_df = df[df.block == "main"]
    off_df = df[df.block == "offline"]

    summ = summary(main_df)
    summ.to_csv(out / "summary.csv", index=False)
    cmp_ = compare_s4(main_df)
    cmp_.to_csv(out / "s4_vs_best.csv", index=False)
    reg = regret(main_df)
    reg.to_csv(out / "regret.csv")
    md = ["# Результаты стенда", "", f"Прогонов основного блока: {len(main_df)}, офлайн-блока: {len(off_df)}.",
          f"Коммит: {df.git_commit.iloc[0]}; Chromium {df.chromium.iloc[0]}; transformers.js {df.transformers.iloc[0]}.", ""]

    md += ["## Медиана p95 по повторам, с [95 % bootstrap-интервал]", "",
           "| Сеть × устройство | " + " | ".join(STRATS) + " |", "|---|" + "---|" * 4]
    for n in NETS:
        for d in DEVS:
            cells = []
            for s in STRATS:
                r = summ[(summ.strategy == s) & (summ.network == n) & (summ.device == d)]
                cells.append("—" if r.empty else f"{fmt_s(r.p95_median.iloc[0])} [{fmt_s(r.p95_ci_low.iloc[0])}; {fmt_s(r.p95_ci_high.iloc[0])}]")
            md.append(f"| {n} × {d} | " + " | ".join(cells) + " |")

    md += ["", "## S4 против лучшей простой стратегии условия", "",
           "| Условие | Лучшая простая | p95 S4, с | p95 лучшей, с | Δ, % [95 % ДИ] | p₁ (Холм) | δ Клиффа | ΔWER, п. п. | H₁ |",
           "|---|---|---|---|---|---|---|---|---|"]
    for _, r in cmp_.iterrows():
        md.append(f"| {r.network} × {r.device} | {r.best_simple} | {fmt_s(r.p95_S4)} | {fmt_s(r.p95_best)} | "
                  f"{r.rel_diff*100:+.0f} [{r.ci_low*100:+.0f}; {r.ci_high*100:+.0f}] | {r.p_holm:.3f} | "
                  f"{r.cliff:+.2f} ({r.effect}) | {r.wer_diff_pp:+.1f} | {'да' if r.h1 else 'нет'} |")

    md += ["", "## Худший случай каждой стратегии (проигрыш лучшей стратегии условия по медиане p95)", "",
           "| Стратегия | Максимальный проигрыш, % | Среднее, % | Условие худшего случая |", "|---|---|---|---|"]
    for s in STRATS:
        col = reg[s]
        md.append(f"| {s} | {col.max()*100:.0f} | {col.mean()*100:.0f} | {' × '.join(col.idxmax())} |")

    if len(off_df):
        md += ["", "## H₂: работа без сети (N0 × D1)", "", "| Стратегия | Прогонов | Доля распознанных |", "|---|---|---|"]
        for s, g in off_df.groupby("strategy"):
            md.append(f"| {s} | {len(g)} | {g.recognized.mean()*100:.0f} % |")

    wer_s = main_df.groupby("strategy")["wer"].median()
    md += ["", "## Качество (WER, медиана по прогонам)", "", "| " + " | ".join(STRATS) + " |", "|" + "---|" * 4,
           "| " + " | ".join(f"{wer_s.get(s, float('nan'))*100:.1f} %" for s in STRATS) + " |"]
    loc = fr[(fr["where"] == "local") & ~fr.failed]
    srv = fr[(fr["where"] == "server") & ~fr.failed]
    md += ["", f"WER всех локально распознанных фрагментов: {wer(loc.ref.tolist(), loc.hyp.tolist())*100:.1f} %, "
           f"на сервере: {wer(srv.ref.tolist(), srv.hyp.tolist())*100:.1f} %."]

    # проверка стенда
    rho, p_rho = plot_stability(main_df, out)
    md += ["", "## Проверка стенда", "",
           f"- Отставание подачи фрагментов от расписания: p95 по прогонам — медиана {main_df.lag_p95_ms.median():.1f} мс, "
           f"максимум {main_df.lag_max_ms.max():.0f} мс.",
           f"- CPU сервера (лимит 1 ядро): среднее за прогон — медиана {main_df.server_cpu_mean.median():.2f}, "
           f"максимум {main_df.server_cpu_mean.max():.2f}; пик по Prometheus — максимум {main_df.server_cpu_peak.max():.2f}.",
           f"- Тренд p95/медиана конфигурации от номера прогона: ρ Спирмена = {rho:+.3f}, p = {p_rho:.3f}.",
           f"- Загрузка локальной модели: медиана {main_df.local_load_ms.median()/1000:.1f} с."]
    aa_df, _ = load(a.raw, a.aa)
    aa = None
    if len(aa_df) >= 4:
        aa = aa_test(aa_df)
        plot_aa(aa, out)
        md += [f"- A/A-тест (S2 × N1 × D1, {aa['n']} прогонов): медианы p95 групп {fmt_s(aa['median_A'])} и "
               f"{fmt_s(aa['median_B'])} с, разница {aa['rel_diff']*100:+.1f} %, Манн–Уитни p = {aa['p']:.2f}, "
               f"δ Клиффа = {aa['cliff']:+.2f}; коэффициент вариации p95 = {aa['cv']*100:.1f} %, "
               f"минимальный обнаружимый эффект ≈ {aa['mde_rel']*100:.1f} %; тренд ρ = {aa['spearman_rho']:+.2f} "
               f"(p = {aa['spearman_p']:.2f})."]
        (out / "aa.json").write_text(json.dumps(aa, ensure_ascii=False, indent=2))

    plot_boxes(main_df, out)
    plot_ecdf(fr[fr.run_id.isin(main_df.run_id)], out)
    plot_heatmap(reg, out)
    plot_timeline(fr, out)
    (out / "stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
