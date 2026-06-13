from __future__ import annotations

import csv
import json
import random
import signal
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.training.logging import ARTIFACT_SCHEMA_VERSION, save_evaluation_artifacts, to_jsonable, write_json
from src.training.loops import evaluate_predictions


FEATURE_DIR = Path(__file__).resolve().parents[2] / "data" / "features" / "fma_medium"
LABEL_MAP_PATH = Path(__file__).resolve().parents[2] / "data" / "splits" / "fma_medium" / "label_map.json"
CLASS_WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "data" / "splits" / "fma_medium" / "class_weights.json"
FEATURE_GROUPS_PATH = FEATURE_DIR / "feature_groups.json"
VALID_REFIT_SCOPES = {"train_only", "train_validation"}
DEFAULT_REFIT_SCOPE = "train_validation"

CLASSICAL_ARTIFACT_NAMES = [
    "config.json",
    "environment.json",
    "grid_search_results.csv",
    "grid_search_results.json",
    "model.joblib",
    "metrics.json",
    "classification_report.json",
    "report.txt",
    "confusion_matrix.npy",
    "confusion_matrix_normalized.npy",
    "top_confusions.json",
    "predictions.npy",
    "probabilities.npy",
    "decision_scores.npy",
    "test_labels.npy",
    "test_track_ids.npy",
    "cka_reference.json",
    "run_summary.json",
    "run_manifest.json",
]

# sigalrm is only available on unix; on windows the grid search runs without
# a wall-clock timeout, which is acceptable since the small-grid fallback is
# a performance convenience, not a correctness requirement.
_HAS_ALARM = hasattr(signal, "SIGALRM")


class FitTimeoutError(RuntimeError):
    pass


def _timeout_handler(signum: int, frame: object) -> None:
    raise FitTimeoutError("svm grid search exceeded wall-clock limit")


@dataclass
class SVMArtifacts:
    model: object
    used_small_grid: bool
    best_params: dict[str, object]
    best_score: float
    fit_seconds: float
    refit_scope: str


@dataclass
class RandomForestArtifacts:
    model: object
    best_params: dict[str, object]
    best_score: float
    fit_seconds: float
    refit_scope: str


@dataclass
class XGBoostArtifacts:
    model: object
    best_params: dict[str, object]
    best_score: float
    fit_seconds: float
    used_cuda: bool
    refit_scope: str


# the cka analysis compares neural penultimate representations against these fixed
# classical feature matrices, not against hidden layers inside tree/svm models.
def classical_cka_reference() -> dict[str, object]:
    return {
        "description": "deep-model representations are compared against the fixed scaled classical feature matrices, not against classical model internals",
        "full_feature_dim": 540,
        "cka_group_dim": 372,
        "feature_dir": str(FEATURE_DIR),
        "feature_groups_path": str(FEATURE_GROUPS_PATH),
        "test_features_path": str(FEATURE_DIR / "test_features.npy"),
        "test_labels_path": str(FEATURE_DIR / "test_labels.npy"),
        "test_track_ids_path": str(FEATURE_DIR / "test_track_ids.npy"),
    }


# load pre-scaled feature arrays and integer labels for all three splits.
# the arrays were already standardized (train-fit scaler applied to all splits)
# during feature-matrix construction, so no further scaling is needed here.
def load_split_arrays() -> dict[str, np.ndarray]:
    arrays = {}
    for split in ["training", "validation", "test"]:
        arrays[f"{split}_features"] = np.load(FEATURE_DIR / f"{split}_features.npy")
        arrays[f"{split}_labels"] = np.load(FEATURE_DIR / f"{split}_labels.npy")
    return arrays


def load_class_weights() -> dict[int, float]:
    payload = json.loads(CLASS_WEIGHTS_PATH.read_text(encoding="utf-8"))
    raw_weights = {int(label): float(weight) for label, weight in payload["weights_by_id"].items()}
    training_labels = np.load(FEATURE_DIR / "training_labels.npy")
    training_weights = np.asarray([raw_weights[int(label)] for label in training_labels], dtype=np.float64)
    normalizer = float(np.mean(training_weights))
    return {label: float(weight / normalizer) for label, weight in raw_weights.items()}


def make_sample_weights(labels: np.ndarray, class_weights: dict[int, float] | None = None) -> np.ndarray:
    weights = class_weights if class_weights is not None else load_class_weights()
    return np.asarray([weights[int(label)] for label in labels], dtype=np.float32)


# we reuse the spectrogram models' label names so saved confusion summaries
# stay comparable across classical and neural runs.
def load_label_names() -> dict[int, str]:
    payload = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    return {int(index): str(label) for index, label in payload["id_to_label"].items()}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_classical_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def validate_refit_scope(refit_scope: str) -> str:
    if refit_scope not in VALID_REFIT_SCOPES:
        raise ValueError(f"refit_scope must be one of {sorted(VALID_REFIT_SCOPES)}, found {refit_scope!r}")
    return refit_scope


def final_recommendation(refit_scope: str = DEFAULT_REFIT_SCOPE) -> dict[str, str]:
    return {
        "default_refit_scope": DEFAULT_REFIT_SCOPE,
        "selected_refit_scope": refit_scope,
        "recommendation": "Use train_validation for final benchmark runs after validation has selected hyperparameters; use train_only only for strict validation-is-never-refit ablations.",
    }


def refit_arrays(arrays: dict[str, np.ndarray], refit_scope: str) -> tuple[np.ndarray, np.ndarray]:
    validate_refit_scope(refit_scope)
    if refit_scope == "train_only":
        return arrays["training_features"], arrays["training_labels"]
    return (
        np.concatenate([arrays["training_features"], arrays["validation_features"]], axis=0),
        np.concatenate([arrays["training_labels"], arrays["validation_labels"]], axis=0),
    )


def write_classical_run_manifest(output_dir: Path, payload: dict[str, Any]) -> None:
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "updated_at_utc": utc_now(),
        "artifact_paths": {name: str(output_dir / name) for name in CLASSICAL_ARTIFACT_NAMES},
        **payload,
    }
    write_json(output_dir / "run_manifest.json", manifest)


def predict_probabilities_if_available(model: object, inputs: np.ndarray) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    try:
        return np.asarray(model.predict_proba(inputs), dtype=np.float64)
    except AttributeError:
        return None


def decision_scores_if_available(model: object, inputs: np.ndarray) -> np.ndarray | None:
    if not hasattr(model, "decision_function"):
        return None
    try:
        return np.asarray(model.decision_function(inputs), dtype=np.float64)
    except AttributeError:
        return None


def _predefined_train_validation_split(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, PredefinedSplit]:
    train_x = arrays["training_features"]
    val_x = arrays["validation_features"]
    train_y = arrays["training_labels"]
    val_y = arrays["validation_labels"]

    stacked_x = np.concatenate([train_x, val_x], axis=0)
    stacked_y = np.concatenate([train_y, val_y], axis=0)
    test_fold = np.concatenate(
        [
            np.full(shape=train_y.shape[0], fill_value=-1, dtype=np.int64),
            np.zeros(shape=val_y.shape[0], dtype=np.int64),
        ]
    )
    return stacked_x, stacked_y, PredefinedSplit(test_fold=test_fold)


def _cv_value(value: Any) -> Any:
    if np.ma.is_masked(value):
        return None
    return to_jsonable(value)


def _cv_results_rows(cv_results: dict[str, Any]) -> list[dict[str, Any]]:
    row_count = len(cv_results["params"])
    rows: list[dict[str, Any]] = []
    for row_index in range(row_count):
        row: dict[str, Any] = {}
        for key, values in cv_results.items():
            if key == "params":
                row[key] = values[row_index]
            else:
                row[key] = _cv_value(values[row_index])
        rows.append(row)
    return rows


def save_grid_search_results(search: GridSearchCV, output_dir: Path) -> None:
    rows = _cv_results_rows(search.cv_results_)
    write_json(output_dir / "grid_search_results.json", {"results": rows})
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with (output_dir / "grid_search_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(to_jsonable(value), sort_keys=True) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


# evaluate any classical estimator on the held-out test split and persist the
# same artifact shape used by neural runs. probability-derived metrics are
# included automatically for estimators such as random forest and xgboost.
def evaluate_classical_model(model: object, output_dir: Path, model_name: str) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_split_arrays()
    test_x = arrays["test_features"]
    test_y = arrays["test_labels"]
    predictions = np.asarray(model.predict(test_x), dtype=np.int64)
    probabilities = predict_probabilities_if_available(model, test_x)
    decision_scores = decision_scores_if_available(model, test_x)

    evaluation = evaluate_predictions(
        test_y.tolist(),
        predictions.tolist(),
        labels=list(range(16)),
        probabilities=probabilities.tolist() if probabilities is not None else None,
        label_names=load_label_names(),
    )
    evaluation["model_name"] = model_name
    evaluation["probability_metrics_available"] = probabilities is not None
    evaluation["decision_scores_available"] = decision_scores is not None
    evaluation["classical_cka_reference"] = classical_cka_reference()

    joblib.dump(model, output_dir / "model.joblib")
    np.save(output_dir / "predictions.npy", predictions, allow_pickle=False)
    np.save(output_dir / "test_labels.npy", test_y, allow_pickle=False)
    np.save(output_dir / "test_track_ids.npy", np.load(FEATURE_DIR / "test_track_ids.npy"), allow_pickle=False)
    if probabilities is not None:
        np.save(output_dir / "probabilities.npy", probabilities, allow_pickle=False)
    if decision_scores is not None:
        np.save(output_dir / "decision_scores.npy", decision_scores, allow_pickle=False)
    save_evaluation_artifacts(output_dir, evaluation)
    write_json(output_dir / "cka_reference.json", classical_cka_reference())
    write_json(output_dir / "metrics.json", evaluation)
    return evaluation


# wrap svc in a pipeline so gridsearchcv can use the svm__ prefix convention.
# we set class_weight="balanced" so sklearn weights classes by inverse
# frequency, which we need given the extreme genre imbalance.
def build_svm_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "svm",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                ),
            )
        ]
    )


def build_random_forest_estimator(n_jobs: int = -1, seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        class_weight="balanced_subsample",
        n_jobs=n_jobs,
        random_state=seed,
    )


def build_xgboost_estimator(use_cuda: bool = False, n_jobs: int = -1, seed: int = 42) -> object:
    from xgboost import XGBClassifier

    kwargs: dict[str, object] = {
        "objective": "multi:softprob",
        "num_class": 16,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": n_jobs,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    }
    if use_cuda:
        kwargs["device"] = "cuda"
    return XGBClassifier(**kwargs)


def run_random_forest_grid_search(
    output_dir: Path | None = None,
    param_grid: dict[str, list[object]] | None = None,
    estimator_n_jobs: int = -1,
    verbose: int = 2,
    seed: int = 42,
    refit_scope: str = DEFAULT_REFIT_SCOPE,
) -> RandomForestArtifacts:
    set_classical_seed(seed)
    arrays = load_split_arrays()
    stacked_x, stacked_y, split = _predefined_train_validation_split(arrays)
    grid = param_grid if param_grid is not None else {"n_estimators": [200, 500, 1000], "max_depth": [20, 50, None]}

    search = GridSearchCV(
        estimator=build_random_forest_estimator(n_jobs=estimator_n_jobs, seed=seed),
        param_grid=grid,
        scoring="f1_macro",
        cv=split,
        n_jobs=1,
        verbose=verbose,
        return_train_score=True,
        refit=False,
    )
    start_time = time.perf_counter()
    search.fit(stacked_x, stacked_y)
    final_x, final_y = refit_arrays(arrays, refit_scope)
    model = build_random_forest_estimator(n_jobs=estimator_n_jobs, seed=seed)
    model.set_params(**search.best_params_)
    model.fit(final_x, final_y)
    fit_seconds = time.perf_counter() - start_time
    if output_dir is not None:
        save_grid_search_results(search, output_dir)
    return RandomForestArtifacts(
        model=model,
        best_params=dict(search.best_params_),
        best_score=float(search.best_score_),
        fit_seconds=fit_seconds,
        refit_scope=refit_scope,
    )


def run_xgboost_grid_search(
    output_dir: Path | None = None,
    param_grid: dict[str, list[object]] | None = None,
    use_cuda: bool = False,
    estimator_n_jobs: int = -1,
    verbose: int = 2,
    seed: int = 42,
    refit_scope: str = DEFAULT_REFIT_SCOPE,
) -> XGBoostArtifacts:
    set_classical_seed(seed)
    arrays = load_split_arrays()
    stacked_x, stacked_y, split = _predefined_train_validation_split(arrays)
    class_weights = load_class_weights()
    sample_weights = make_sample_weights(stacked_y, class_weights)
    grid = (
        param_grid
        if param_grid is not None
        else {
            "learning_rate": [0.01, 0.1, 0.3],
            "max_depth": [3, 6, 9],
            "n_estimators": [100, 500, 1000],
        }
    )

    search = GridSearchCV(
        estimator=build_xgboost_estimator(use_cuda=use_cuda, n_jobs=estimator_n_jobs, seed=seed),
        param_grid=grid,
        scoring="f1_macro",
        cv=split,
        n_jobs=1,
        verbose=verbose,
        return_train_score=True,
        refit=False,
    )
    start_time = time.perf_counter()
    search.fit(stacked_x, stacked_y, sample_weight=sample_weights)
    final_x, final_y = refit_arrays(arrays, refit_scope)
    model = build_xgboost_estimator(use_cuda=use_cuda, n_jobs=estimator_n_jobs, seed=seed)
    model.set_params(**search.best_params_)
    model.fit(final_x, final_y, sample_weight=make_sample_weights(final_y, class_weights))
    fit_seconds = time.perf_counter() - start_time
    if output_dir is not None:
        save_grid_search_results(search, output_dir)
    return XGBoostArtifacts(
        model=model,
        best_params=dict(search.best_params_),
        best_score=float(search.best_score_),
        fit_seconds=fit_seconds,
        used_cuda=use_cuda,
        refit_scope=refit_scope,
    )


# run a hyperparameter grid search for the rbf svm. uses a predefined split
# (not k-fold) so the fixed train/validation boundary is respected.
# if the full grid exceeds the timeout, falls back to a smaller grid.
def run_grid_search(
    timeout_seconds: int = 900,
    output_dir: Path | None = None,
    param_grid: dict[str, list[object]] | None = None,
    n_jobs: int = 1,
    verbose: int = 2,
    seed: int = 42,
    refit_scope: str = DEFAULT_REFIT_SCOPE,
) -> SVMArtifacts:
    set_classical_seed(seed)
    arrays = load_split_arrays()
    stacked_x, stacked_y, split = _predefined_train_validation_split(arrays)

    full_grid = param_grid if param_grid is not None else {"svm__C": [0.1, 1, 10, 100], "svm__gamma": ["scale", "auto", 1e-3, 1e-2]}
    small_grid = {"svm__C": [1, 10], "svm__gamma": ["scale", 1e-2]}

    def fit_search(grid: dict[str, list[object]]) -> GridSearchCV:
        search = GridSearchCV(
            estimator=build_svm_pipeline(),
            param_grid=grid,
            scoring="f1_macro",
            cv=split,
            n_jobs=n_jobs,
            verbose=verbose,
            refit=False,
        )
        search.fit(stacked_x, stacked_y)
        return search

    start_time = time.perf_counter()
    if _HAS_ALARM:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)
    try:
        search = fit_search(full_grid)
        used_small_grid = False
        if _HAS_ALARM:
            signal.alarm(0)
    except FitTimeoutError:
        if _HAS_ALARM:
            signal.alarm(0)
        search = fit_search(small_grid)
        used_small_grid = True

    final_x, final_y = refit_arrays(arrays, refit_scope)
    model = build_svm_pipeline()
    model.set_params(**search.best_params_)
    model.fit(final_x, final_y)
    fit_seconds = time.perf_counter() - start_time
    if output_dir is not None:
        save_grid_search_results(search, output_dir)
    return SVMArtifacts(
        model=model,
        used_small_grid=used_small_grid,
        best_params=dict(search.best_params_),
        best_score=float(search.best_score_),
        fit_seconds=fit_seconds,
        refit_scope=refit_scope,
    )


# evaluate the best svm on the held-out test split and persist all artifacts
def evaluate_svm(model: object, output_dir: Path) -> dict[str, object]:
    return evaluate_classical_model(model=model, output_dir=output_dir, model_name="svm_rbf")


def _augment_classical_metrics(
    metrics: dict[str, object],
    output_dir: Path,
    best_params: dict[str, object],
    validation_macro_f1: float,
    fit_seconds: float,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    arrays = load_split_arrays()
    metrics.update(
        {
            "selected_hyperparameters": best_params,
            "validation_macro_f1": float(validation_macro_f1),
            "fit_seconds": float(fit_seconds),
            "sklearn_version": sklearn.__version__,
            "feature_matrix_shape": {
                "training": list(arrays["training_features"].shape),
                "validation": list(arrays["validation_features"].shape),
                "test": list(arrays["test_features"].shape),
            },
        }
    )
    if extra:
        metrics.update(extra)
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def evaluate_random_forest(
    model: object,
    output_dir: Path,
    best_params: dict[str, object],
    validation_macro_f1: float,
    fit_seconds: float,
) -> dict[str, object]:
    metrics = evaluate_classical_model(model=model, output_dir=output_dir, model_name="random_forest")
    return _augment_classical_metrics(
        metrics=metrics,
        output_dir=output_dir,
        best_params=best_params,
        validation_macro_f1=validation_macro_f1,
        fit_seconds=fit_seconds,
    )


def evaluate_xgboost(
    model: object,
    output_dir: Path,
    best_params: dict[str, object],
    validation_macro_f1: float,
    fit_seconds: float,
    used_cuda: bool,
) -> dict[str, object]:
    import xgboost

    metrics = evaluate_classical_model(model=model, output_dir=output_dir, model_name="xgboost")
    return _augment_classical_metrics(
        metrics=metrics,
        output_dir=output_dir,
        best_params=best_params,
        validation_macro_f1=validation_macro_f1,
        fit_seconds=fit_seconds,
        extra={
            "xgboost_version": xgboost.__version__,
            "used_cuda": bool(used_cuda),
            "xgboost_device": "cuda" if used_cuda else "cpu",
        },
    )
