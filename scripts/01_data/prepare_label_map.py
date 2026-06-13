from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.features import write_label_artifacts


def main() -> None:
    label_to_id, _, inverse_frequency = write_label_artifacts()
    ordered_labels = sorted(label_to_id.items(), key=lambda item: item[1])

    print("classes:")
    for label, index in ordered_labels:
        print(f"  {index}: {label}")

    print("weights:")
    for label, index in ordered_labels:
        print(f"  {label}: {inverse_frequency[index]:.8f}")

    summary = {
        "class_count": len(label_to_id),
        "labels": [label for label, _ in ordered_labels],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
