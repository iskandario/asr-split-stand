"""Тесты функций анализа и раннера: uv run pytest -q"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "analysis"), str(ROOT / "runner")]

import analyze  # noqa: E402
import run_experiment  # noqa: E402


def test_normalize_numbers_and_punctuation():
    assert analyze.normalize("Ёлка, 60000 тенге!") == "елка шестьдесят тысяч тенге"


def test_wer_identical_and_empty_hypothesis():
    assert analyze.wer(["а б в"], ["а б в"]) == 0
    assert analyze.wer(["а б"], [""]) == 1.0


def test_cliffs_delta_extremes():
    assert analyze.cliffs_delta([1, 2], [3, 4]) == -1
    assert analyze.cliffs_delta([3, 4], [1, 2]) == 1
    assert analyze.cliffs_label(0.1) == "пренебрежимый"
    assert analyze.cliffs_label(-0.5) == "большой"


def test_holm_matches_manual():
    adj = analyze.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_bootstrap_rel_diff_contains_point():
    a, b = np.array([8, 9, 10, 11, 12.0]), np.array([10, 11, 12, 13, 14.0])
    point, lo, hi = analyze.boot_ci_rel_diff(a, b)
    assert lo <= point <= hi
    assert abs(point - (10 - 12) / 12) < 1e-9


def test_plan_is_full_factorial_and_reproducible():
    cfg = run_experiment.load_config(ROOT / "configs" / "experiment.yaml")
    p1, p2 = run_experiment.build_plan(cfg), run_experiment.build_plan(cfg)
    assert [p["run_id"] for p in p1] == [p["run_id"] for p in p2]
    main = [p for p in p1 if p["block"] == "main"]
    assert len(main) == 4 * 4 * 3 * cfg["blocks"][0]["repetitions"]
    assert len({(p["strategy"], p["network"], p["device"]) for p in main}) == 48


def test_run_spec_uses_fixed_fragments():
    cfg = run_experiment.load_config(ROOT / "configs" / "experiment.yaml")
    spec = run_experiment.run_spec(cfg, run_experiment.build_plan(cfg)[0])
    assert [f["id"] for f in spec["client"]["fragments"]][:2] == ["test_000", "test_001"]
    assert len(spec["client"]["fragments"]) == cfg["fragments_per_run"]
    assert spec["client"]["warmup"][0]["id"] == "tune_000"


def test_network_n4_alternates_two_profiles():
    cfg = run_experiment.load_config(ROOT / "configs" / "experiment.yaml")
    n4 = run_experiment.network_spec(cfg, "N4")
    assert [p["name"] for p in n4["alternate"]] == ["N1", "N3"]
