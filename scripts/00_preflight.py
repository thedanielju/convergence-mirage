from __future__ import annotations

import json
from pathlib import Path

import joblib
import librosa
import pandas as pd
import sklearn
import torch
import xgboost


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "splits" / "fma_medium" / "benchmark_ready" / "manifest.csv"


def main() -> None:
    result = {
        "torch_cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "librosa_version": librosa.__version__,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "joblib_version": joblib.__version__,
        "manifest_exists": MANIFEST_PATH.exists(),
        "manifest_rows": len(pd.read_csv(MANIFEST_PATH)) if MANIFEST_PATH.exists() else 0,
    }

    if not result["torch_cuda_available"]:
        raise RuntimeError("preflight failed: torch.cuda.is_available() is False")
    if not result["manifest_exists"] or result["manifest_rows"] != 16988:
        raise RuntimeError(f"preflight failed: expected benchmark_ready manifest with 16988 rows, found {result['manifest_rows']}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
