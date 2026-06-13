#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
HEADLINE = RESULTS_DIR / "aggregated" / "headline_metrics.json"
OUT = RESULTS_DIR / "aggregated" / "paired_tests.json"

DEEP = ["cnn2d", "bilstm", "transformer", "mamba1", "mamba2"]
SEEDS = [42, 43, 44, 45, 46]
ALPHA = 0.05


def _macro_f1_vector(headline: dict, model: str) -> np.ndarray:
    per = headline[model]["per_seed"]
    return np.array([per[str(s)]["macro_f1"] for s in SEEDS], dtype=float)


def main() -> int:
    if not HEADLINE.exists():
        print(f"missing {HEADLINE} -- run aggregate_metrics.py first", file=sys.stderr)
        return 1
    headline = json.loads(HEADLINE.read_text(encoding="utf-8"))

    pairs = list(combinations(DEEP, 2))
    n_pairs = len(pairs)
    bonf_threshold = ALPHA / n_pairs

    results = []
    for a, b in pairs:
        va = _macro_f1_vector(headline, a)
        vb = _macro_f1_vector(headline, b)
        diffs = va - vb
        if np.allclose(diffs, 0.0):
            test_used = "degenerate_zero_diffs"
            statistic = 0.0
            p_raw = 1.0
        else:
            sw_stat, sw_p = stats.shapiro(diffs)
            if sw_p > 0.05:
                t_stat, p_raw = stats.ttest_rel(va, vb)
                statistic = float(t_stat)
                test_used = "paired_t"
            else:
                try:
                    w_stat, p_raw = stats.wilcoxon(va, vb, zero_method="wilcox", alternative="two-sided")
                    statistic = float(w_stat)
                    test_used = "wilcoxon"
                except ValueError:
                    t_stat, p_raw = stats.ttest_rel(va, vb)
                    statistic = float(t_stat)
                    test_used = "paired_t_fallback"
        p_raw = float(p_raw)
        p_bonf = float(min(1.0, p_raw * n_pairs))
        sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        mean_diff = float(np.mean(diffs))
        cohens_d = float(mean_diff / sd) if sd > 0 else 0.0
        sem = sd / np.sqrt(len(diffs)) if sd > 0 else 0.0
        tcrit = float(stats.t.ppf(0.975, df=len(diffs) - 1)) if len(diffs) > 1 else 0.0
        ci_lo = mean_diff - tcrit * sem
        ci_hi = mean_diff + tcrit * sem

        results.append({
            "pair": f"{a}_vs_{b}",
            "model_a": a,
            "model_b": b,
            "n": int(len(diffs)),
            "macro_f1_a": [float(x) for x in va],
            "macro_f1_b": [float(x) for x in vb],
            "diffs": [float(x) for x in diffs],
            "test_used": test_used,
            "statistic": statistic,
            "p_raw": p_raw,
            "p_bonferroni": p_bonf,
            "significant_after_correction": bool(p_raw < bonf_threshold),
            "cohens_d": cohens_d,
            "mean_diff": mean_diff,
            "ci95": [float(ci_lo), float(ci_hi)],
        })

    significant = [r for r in results if r["significant_after_correction"]]
    payload = {
        "alpha": ALPHA,
        "n_pairs": n_pairs,
        "bonferroni_threshold_raw_p": bonf_threshold,
        "seeds": SEEDS,
        "metric": "macro_f1",
        "pairs": results,
        "summary": {
            "n_significant": len(significant),
            "significant_pairs": [
                {
                    "pair": r["pair"],
                    "test_used": r["test_used"],
                    "p_raw": r["p_raw"],
                    "p_bonferroni": r["p_bonferroni"],
                    "mean_diff": r["mean_diff"],
                    "cohens_d": r["cohens_d"],
                }
                for r in significant
            ],
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")

    print(f"\nPaired tests on macro F1, n=5 seeds, Bonferroni threshold raw p < {bonf_threshold:.4f}")
    print(f"{'pair':<28}{'test':<12}{'p_raw':>10}{'p_bonf':>10}{'d':>8}{'mdiff':>10}{'sig':>5}")
    print("-" * 83)
    for r in sorted(results, key=lambda x: x["p_raw"]):
        print(f"{r['pair']:<28}{r['test_used']:<12}{r['p_raw']:>10.4f}{r['p_bonferroni']:>10.4f}"
              f"{r['cohens_d']:>8.2f}{r['mean_diff']:>10.4f}{('*' if r['significant_after_correction'] else ''):>5}")

    print(f"\nSignificant after Bonferroni: {len(significant)}")
    for r in significant:
        print(f"  {r['pair']}: p_raw={r['p_raw']:.4g} p_bonf={r['p_bonferroni']:.4g} d={r['cohens_d']:.2f} mean_diff={r['mean_diff']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
