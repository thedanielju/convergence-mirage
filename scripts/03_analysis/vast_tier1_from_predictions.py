from __future__ import annotations
import csv, json, math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None

ROOT = Path('results')
OUT_REL = ROOT / 'tier1_reliability'
OUT_IMB = ROOT / 'tier1_class_imbalance'
OUT_REL.mkdir(parents=True, exist_ok=True)
OUT_IMB.mkdir(parents=True, exist_ok=True)

RUN_ROOTS = [
    ROOT / 'long_sequence',
    ROOT / 'ablation' / 'fma_full60_crop15',
    ROOT / 'specaugment_ablation',
]


def load_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def find_array(run: Path, names: list[str]) -> np.ndarray | None:
    candidates = []
    for name in names:
        candidates.extend(run.rglob(name))
    for p in candidates:
        try:
            return np.load(p)
        except Exception:
            pass
    return None


def run_id(run: Path) -> str:
    parts = run.parts
    if 'long_sequence' in parts:
        return 'long_sequence__' + run.name
    if 'specaugment_ablation' in parts:
        return 'specaugment_ablation__' + run.name
    if 'fma_full60_crop15' in parts:
        return 'ablation_fma_full60_crop15__' + run.name
    return run.name

summary = []
imb_rows = []
for root in RUN_ROOTS:
    if not root.exists():
        continue
    for manifest in sorted(root.rglob('run_manifest.json')):
        run = manifest.parent
        rid = run_id(run)
        mj = load_json(manifest) or {}
        probs = find_array(run, ['probabilities.npy', 'test_probabilities.npy', 'y_prob.npy', 'probs.npy'])
        labels = find_array(run, ['labels.npy', 'test_labels.npy', 'y_true.npy'])
        preds = find_array(run, ['predictions.npy', 'test_predictions.npy', 'y_pred.npy'])
        if preds is None and probs is not None:
            preds = np.argmax(probs, axis=1)
        reliability = None
        if probs is not None and labels is not None and len(probs) == len(labels):
            conf = np.max(probs, axis=1)
            pred = np.argmax(probs, axis=1)
            labels_i = labels.astype(int)
            correct = (pred == labels_i).astype(float)
            bins = np.linspace(0.0, 1.0, 11)
            rows = []
            ece = 0.0
            for i in range(10):
                lo, hi = bins[i], bins[i+1]
                mask = (conf >= lo) & (conf < hi if i < 9 else conf <= hi)
                n = int(mask.sum())
                acc = float(correct[mask].mean()) if n else math.nan
                cmean = float(conf[mask].mean()) if n else math.nan
                if n:
                    ece += (n / len(conf)) * abs(acc - cmean)
                rows.append({'bin_low': lo, 'bin_high': hi, 'n': n, 'accuracy': acc, 'confidence': cmean})
            csv_path = OUT_REL / f'{rid}_reliability.csv'
            with csv_path.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            if plt is not None:
                xs = [(r['bin_low'] + r['bin_high']) / 2 for r in rows]
                ys = [0 if math.isnan(r['accuracy']) else r['accuracy'] for r in rows]
                cs = [0 if math.isnan(r['confidence']) else r['confidence'] for r in rows]
                fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
                ax.plot([0, 1], [0, 1], '--', color='0.6', linewidth=1)
                ax.plot(cs, ys, marker='o')
                ax.set_xlabel('Mean confidence')
                ax.set_ylabel('Accuracy')
                ax.set_title(rid.replace('__', ' / '))
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                fig.tight_layout()
                fig.savefig(OUT_REL / f'{rid}_reliability.png')
                plt.close(fig)
            reliability = {'ece_10bin': ece, 'csv': str(csv_path)}
        report_json = None
        for p in list(run.rglob('classification_report.json')) + list(run.rglob('report.json')):
            report_json = load_json(p)
            if isinstance(report_json, dict):
                break
        if isinstance(report_json, dict):
            for key, val in report_json.items():
                if isinstance(val, dict) and 'support' in val:
                    try:
                        imb_rows.append({
                            'run_id': rid,
                            'class': key,
                            'precision': val.get('precision'),
                            'recall': val.get('recall'),
                            'f1_score': val.get('f1-score', val.get('f1_score')),
                            'support': val.get('support'),
                        })
                    except Exception:
                        pass
        elif labels is not None and preds is not None and len(labels) == len(preds):
            for cls in sorted(set(labels.astype(int).tolist())):
                mask = labels.astype(int) == cls
                support = int(mask.sum())
                tp = int(((preds.astype(int) == cls) & mask).sum())
                pred_count = int((preds.astype(int) == cls).sum())
                recall = tp / support if support else math.nan
                precision = tp / pred_count if pred_count else math.nan
                f1 = 2*precision*recall/(precision+recall) if precision == precision and recall == recall and precision+recall else math.nan
                imb_rows.append({'run_id': rid, 'class': str(cls), 'precision': precision, 'recall': recall, 'f1_score': f1, 'support': support})
        summary.append({
            'run_id': rid,
            'path': str(run),
            'status': mj.get('status'),
            'model': mj.get('model') or mj.get('arch'),
            'seed': mj.get('seed'),
            'crop_seconds': mj.get('crop_seconds'),
            'has_probabilities': probs is not None,
            'has_labels': labels is not None,
            'has_predictions': preds is not None,
            'reliability': reliability,
        })

(OUT_REL / 'manifest.json').write_text(json.dumps(summary, indent=2))
if imb_rows:
    out = OUT_IMB / 'per_class_metrics.csv'
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(imb_rows[0].keys()))
        w.writeheader(); w.writerows(imb_rows)
    (OUT_IMB / 'manifest.json').write_text(json.dumps({'rows': len(imb_rows), 'csv': str(out)}, indent=2))
else:
    (OUT_IMB / 'manifest.json').write_text(json.dumps({'rows': 0, 'note': 'No classification report or label/prediction arrays found in exported runs.'}, indent=2))
print(json.dumps({'reliability_runs': len(summary), 'class_rows': len(imb_rows)}, indent=2))
