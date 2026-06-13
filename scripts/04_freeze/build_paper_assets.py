from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


HEADLINE_PATH = REPO_ROOT / "results" / "aggregated" / "headline_metrics.json"
PAIRED_PATH = REPO_ROOT / "results" / "aggregated" / "paired_tests.json"
AGG_CM_PATH = REPO_ROOT / "results" / "aggregated" / "per_model_aggregate_confusion.npz"
CKA_DIR = REPO_ROOT / "results" / "cka"
ERROR_DIR = REPO_ROOT / "results" / "error_analysis"
LABEL_MAP_PATH = REPO_ROOT / "data" / "splits" / "fma_medium" / "label_map.json"
LEARNING_CURVES_DIR = REPO_ROOT / "results" / "learning_curves"
ABLATION_MANIFEST = REPO_ROOT / "results" / "ablation_manifest.json"
LONG_SEQ_DIR = REPO_ROOT / "results" / "long_sequence"
OUT_DIR = REPO_ROOT / "results" / "paper_assets"

ALL_MODELS = ("cnn2d", "bilstm", "transformer", "mamba1", "mamba2", "svm_rbf", "random_forest", "xgboost")
DEEP_MODELS = ("cnn2d", "bilstm", "transformer", "mamba1", "mamba2")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str] | None = None) -> None:
    import csv

    if columns is None and rows:
        columns = list(rows[0])
    columns = columns or []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def build_table1(headline: dict[str, Any]) -> list[dict[str, Any]]:
    metric_keys = ["macro_f1", "accuracy", "weighted_f1", "balanced_accuracy", "macro_precision", "macro_recall", "kappa", "mcc"]
    aliases = {"balanced_accuracy": "balanced_acc"}
    rows: list[dict[str, Any]] = []
    for model in ALL_MODELS:
        if model not in headline:
            continue
        m = headline[model].get("mean", {})
        s = headline[model].get("std", {})
        row: dict[str, Any] = {"model": model}
        for mk in metric_keys:
            # the aggregate writer uses a couple of legacy metric names, so we fall back
            # to the alias when the canonical key is absent rather than dropping the cell.
            source_key = mk if mk in m else aliases.get(mk, mk)
            mean_v = float(m.get(source_key, float("nan")))
            std_v = float(s.get(source_key, float("nan")))
            row[mk] = fmt_mean_std(mean_v, std_v)
        rows.append(row)
    return rows


def build_table2(headline: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in DEEP_MODELS:
        if model not in headline:
            continue
        m = headline[model].get("mean", {})
        s = headline[model].get("std", {})
        row: dict[str, Any] = {
            "model": model,
            "top2_acc": fmt_mean_std(m.get("top2_accuracy", m.get("top2_acc", float("nan"))), s.get("top2_accuracy", s.get("top2_acc", float("nan")))),
            "top3_acc": fmt_mean_std(m.get("top3_accuracy", m.get("top3_acc", float("nan"))), s.get("top3_accuracy", s.get("top3_acc", float("nan")))),
            "ece": fmt_mean_std(m.get("ece", float("nan")), s.get("ece", float("nan"))),
            "brier": fmt_mean_std(m.get("brier", float("nan")), s.get("brier", float("nan"))),
            "mean_confidence": fmt_mean_std(m.get("mean_confidence", float("nan")), s.get("mean_confidence", float("nan"))),
            "mean_entropy": fmt_mean_std(m.get("mean_entropy", float("nan")), s.get("mean_entropy", float("nan"))),
        }
        rows.append(row)
    return rows


def build_table3(error_top_pairs: dict[str, Any], distances: dict[str, Any]) -> list[dict[str, Any]]:
    if not error_top_pairs or not distances:
        return []
    pair_lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for entry in distances.get("pairs", []):
        i, j = int(entry["i"]), int(entry["j"])
        pair_lookup[(min(i, j), max(i, j))] = entry
    rows: list[dict[str, Any]] = []
    for arch in DEEP_MODELS:
        pairs = error_top_pairs.get(arch, [])
        for rank, p in enumerate(pairs[:5], start=1):
            i, j = int(p["i"]), int(p["j"])
            entry = pair_lookup.get((min(i, j), max(i, j)), {})
            row = {
                "arch": arch,
                "rank": rank,
                "pair": f"{p.get('i_name', i)} ↔ {p.get('j_name', j)}",
                "sum_off_diag_normalized": f"{p['sum_off_diag_normalized']:.4f}",
                "full_distance": f"{entry.get('full_distance', float('nan')):.3f}",
            }
            for g, d in entry.get("per_group_distance", {}).items():
                row[f"d_{g}"] = f"{d:.3f}"
            rows.append(row)
    return rows


def build_ablation_table(payload: Any) -> list[dict[str, Any]]:
    if not payload:
        return []
    entries = payload.get("entries") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return []
    rows: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        row = {
            "model": entry.get("model_key") or entry.get("model") or "",
            "seed": entry.get("seed") or "",
            "crop_seconds": entry.get("crop_seconds") or "",
            "status": entry.get("status") or entry.get("validation_status") or "",
            "macro_f1": entry.get("macro_f1") or metrics.get("test_macro_f1") or metrics.get("macro_f1") or "",
            "accuracy": entry.get("accuracy") or metrics.get("test_accuracy") or metrics.get("accuracy") or "",
            "output_dir": entry.get("output_dir") or entry.get("output_dir_repo_relative") or entry.get("path") or "",
        }
        rows.append(row)
    return rows


def build_long_sequence_table(payload: Any | None = None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if payload is None:
        manifest_path = LONG_SEQ_DIR / "manifest.json"
        payload = load_json(manifest_path)
    if isinstance(payload, dict) and isinstance(payload.get("entries"), list):
        entries = [entry for entry in payload["entries"] if isinstance(entry, dict)]
    elif isinstance(payload, list):
        entries = [entry for entry in payload if isinstance(entry, dict)]
    elif LONG_SEQ_DIR.exists():
        for manifest_path in sorted(LONG_SEQ_DIR.glob("*/run_manifest.json")):
            run_dir = manifest_path.parent
            manifest = load_json(manifest_path) or {}
            metrics = load_json(run_dir / "metrics.json") or {}
            entries.append(
                {
                    "model_key": manifest.get("model") or "",
                    "seed": manifest.get("seed") or "",
                    "crop_seconds": manifest.get("crop_seconds") or "",
                    "status": manifest.get("status") or "",
                    "macro_f1": metrics.get("test_macro_f1") or metrics.get("macro_f1") or "",
                    "accuracy": metrics.get("test_accuracy") or metrics.get("accuracy") or "",
                    "output_dir": str(run_dir.relative_to(REPO_ROOT)),
                    "data_root": manifest.get("data_root") or "",
                }
            )
    rows: list[dict[str, Any]] = []
    for entry in entries:
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        rows.append(
            {
                "model": entry.get("model_key") or entry.get("model") or "",
                "seed": entry.get("seed") or "",
                "crop_seconds": entry.get("crop_seconds") or "",
                "status": entry.get("status") or "",
                "macro_f1": entry.get("macro_f1") or metrics.get("test_macro_f1") or metrics.get("macro_f1") or "",
                "accuracy": entry.get("accuracy") or metrics.get("test_accuracy") or metrics.get("accuracy") or "",
                "data_root": entry.get("data_root") or "",
                "output_dir": entry.get("output_dir") or entry.get("path") or "",
            }
        )
    return rows


def fig_inter_arch_cka(out_path: Path) -> None:
    arr_path = CKA_DIR / "inter_arch_linear.npy"
    labels_path = CKA_DIR / "inter_arch_linear_labels.json"
    # we skip a figure rather than crash when its upstream CKA outputs are missing, so a
    # partial rebuild still produces every asset it can.
    if not arr_path.exists() or not labels_path.exists():
        return
    arr = np.load(arr_path)
    names = json.loads(labels_path.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(arr, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", color="white" if arr[i, j] < 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="linear CKA")
    ax.set_title("Inter-architecture linear CKA (penultimate, seed 42)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_8group_cka_profile(out_path: Path) -> None:
    p = CKA_DIR / "arch_to_classical_groups.json"
    if not p.exists():
        return
    payload = json.loads(p.read_text(encoding="utf-8"))
    matrix = np.array(payload["matrix"], dtype=np.float64)
    archs = payload["model_keys"]
    groups = payload["group_names"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(groups))
    width = 0.16
    for k, arch in enumerate(archs):
        ax.bar(x + (k - (len(archs) - 1) / 2) * width, matrix[k], width=width, label=arch)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel("linear CKA")
    ax.set_title("Architecture vs classical-feature group CKA")
    ax.legend(fontsize=8, ncol=3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_cka_vs_f1(out_path: Path, headline: dict[str, Any]) -> None:
    p = CKA_DIR / "arch_to_classical_full.json"
    if not p.exists():
        return
    payload = json.loads(p.read_text(encoding="utf-8"))
    full = payload["per_arch_full_linear_cka"]
    pts = []
    for arch, cka in full.items():
        m = headline.get(arch, {}).get("mean", {})
        f1 = float(m.get("macro_f1", float("nan")))
        if np.isfinite(f1):
            pts.append((arch, float(cka), f1))
    fig, ax = plt.subplots(figsize=(5, 4))
    for arch, cka, f1 in pts:
        ax.scatter(cka, f1, s=70)
        ax.annotate(arch, (cka, f1), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("CKA(arch, classical 540-d)")
    ax.set_ylabel("Macro F1 (mean across 5 seeds)")
    ax.set_title("Classical-alignment CKA vs macro F1")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_novelty_gap(out_path: Path) -> None:
    p = CKA_DIR / "novelty_gap.json"
    if not p.exists():
        return
    payload = json.loads(p.read_text(encoding="utf-8"))
    archs = list(payload)
    gaps = [payload[a]["novelty_gap"] for a in archs]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(archs, gaps, color="#2b8cbe")
    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, gap, f"{gap:+.3f}", ha="center", va="bottom" if gap >= 0 else "top", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("inter-arch CKA − classical CKA")
    ax.set_title("Novelty gap per architecture")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_confusion_matrices(out_path_dir: Path) -> None:
    if not AGG_CM_PATH.exists() or not LABEL_MAP_PATH.exists():
        return
    out_path_dir.mkdir(parents=True, exist_ok=True)
    label_payload = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    id_to_label = {int(k): str(v) for k, v in label_payload["id_to_label"].items()}
    n = max(id_to_label) + 1
    labels = [id_to_label.get(i, str(i)) for i in range(n)]
    arch = np.load(AGG_CM_PATH, allow_pickle=False)
    for key in arch.files:
        cm = arch[key].astype(np.float64, copy=False)
        row_sum = cm.sum(axis=1, keepdims=True)
        norm = np.where(row_sum > 0, cm / row_sum, 0.0)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        for i in range(n):
            for j in range(n):
                v = norm[i, j]
                if v >= 0.05:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.5 else "black", fontsize=6)
        ax.set_title(f"{key} confusion (row-normalized, summed across seeds)")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        fig.tight_layout()
        fig.savefig(out_path_dir / f"confusion_{key}.png", dpi=150)
        plt.close(fig)


def fig_learning_curves(out_path: Path) -> None:
    if not LEARNING_CURVES_DIR.exists():
        return
    aggregated = LEARNING_CURVES_DIR / "aggregated.json"
    if not aggregated.exists():
        return
    payload = json.loads(aggregated.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6, 4))
    for arch, points in payload.items():
        if not isinstance(points, dict):
            continue
        fractions = sorted(int(f) for f in points)
        f1s = [points[str(f)] for f in fractions]
        ax.plot(fractions, f1s, marker="o", label=arch)
    ax.set_xlabel("Training data fraction (%)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Learning curves (seed 42)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figs_dir = OUT_DIR / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir = figs_dir / "confusion_matrices"

    headline = load_json(HEADLINE_PATH) or {}
    paired = load_json(PAIRED_PATH) or {}
    distances = load_json(ERROR_DIR / "inter_genre_distances_per_group.json") or {}
    error_top = load_json(ERROR_DIR / "top_confused_pairs_per_arch.json") or {}

    table1 = build_table1(headline)
    if table1:
        cols = ["model", "macro_f1", "accuracy", "weighted_f1", "balanced_accuracy", "macro_precision", "macro_recall", "kappa", "mcc"]
        write_csv(table1, OUT_DIR / "table1_performance.csv", cols)
        (OUT_DIR / "table1_performance.json").write_text(json.dumps(table1, indent=2), encoding="utf-8")

    table2 = build_table2(headline)
    if table2:
        cols = ["model", "top2_acc", "top3_acc", "ece", "brier", "mean_confidence", "mean_entropy"]
        write_csv(table2, OUT_DIR / "table2_efficiency_calibration.csv", cols)
        (OUT_DIR / "table2_efficiency_calibration.json").write_text(json.dumps(table2, indent=2), encoding="utf-8")

    table3 = build_table3(error_top, distances)
    if table3:
        cols = list(table3[0])
        write_csv(table3, OUT_DIR / "table3_top_confusions.csv", cols)
        (OUT_DIR / "table3_top_confusions.json").write_text(json.dumps(table3, indent=2), encoding="utf-8")

    ablation_table = build_ablation_table(load_json(ABLATION_MANIFEST))
    if ablation_table:
        cols = ["model", "seed", "crop_seconds", "status", "macro_f1", "accuracy", "output_dir"]
        write_csv(ablation_table, OUT_DIR / "table4_ablation.csv", cols)
        (OUT_DIR / "table4_ablation.json").write_text(json.dumps(ablation_table, indent=2), encoding="utf-8")

    long_sequence_table = build_long_sequence_table()
    if long_sequence_table:
        cols = ["model", "seed", "crop_seconds", "status", "macro_f1", "accuracy", "data_root", "output_dir"]
        write_csv(long_sequence_table, OUT_DIR / "table5_long_sequence.csv", cols)
        (OUT_DIR / "table5_long_sequence.json").write_text(json.dumps(long_sequence_table, indent=2), encoding="utf-8")

    fig_inter_arch_cka(figs_dir / "fig1_inter_arch_cka.png")
    fig_8group_cka_profile(figs_dir / "fig2_8group_cka_profile.png")
    fig_cka_vs_f1(figs_dir / "fig3_cka_vs_f1.png", headline)
    fig_novelty_gap(figs_dir / "fig4_novelty_gap.png")
    fig_confusion_matrices(confusion_dir)
    fig_learning_curves(figs_dir / "fig6_learning_curves.png")

    summary = {
        "created_at_utc": utc_now(),
        "tables": sorted(p.name for p in OUT_DIR.iterdir() if p.is_file()),
        "figures": sorted(p.name for p in figs_dir.iterdir() if p.is_file()),
        "confusion_figures": sorted(p.name for p in confusion_dir.iterdir() if p.is_file()) if confusion_dir.exists() else [],
        "paired_tests_significant_after_bonferroni": [pair for pair, body in (paired.items() if isinstance(paired, dict) else []) if isinstance(body, dict) and body.get("significant_after_correction")],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[paper-assets] tables:", summary["tables"])
    print("[paper-assets] figures:", summary["figures"])
    print("[paper-assets] confusion figures:", len(summary["confusion_figures"]))


if __name__ == "__main__":
    main()
