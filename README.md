# The Convergence Mirage

**By:** Daniel Ju

This repository contains the reproducible source code, split metadata, and compact result artifacts for a controlled music genre classification study on FMA Medium. The project benchmarks eight model families, SVM-RBF (Radial Basis Function), Random Forest, XGBoost, 2D-CNN, BiLSTM, Transformer, Mamba-1, and Mamba-2, under a five-seed frozen evaluation protocol, and audits their penultimate-layer representations with Centered Kernel Alignment (CKA).

The primary result is a measurement caution: although trained deep-model CKA is high and seemingly indicates broad representational convergence across architectures, a multi-initialization untrained-floor control explains most of that apparent deep-vs-deep convergence. Still, deep representations remain far from my 540-dimensional classical DSP feature matrix combining 518 FMA-provided audio descriptors with 22 custom rhythm features, revealing a genuine deep-vs-classical novelty gap. I interpret this asymmetry as a "convergence mirage"; hence the name of my working paper.

## Highlights

- Benchmarks 40 frozen runs: 8 model families x 5 random seeds.
- Includes classical DSP baselines, CNN/LSTM/Transformer/Mamba sequence models, paired tests, calibration summaries, CKA, error analysis, and ablation artifacts.
- Separates reproducible source and compact derived artifacts from raw audio, processed spectrogram tensors, checkpoints, and provider-specific logs.
- Treats representation similarity as a measured claim with explicit untrained-floor and deep-vs-classical controls.

## Repository Map

| Path | Description |
|---|---|
| `src/` | Core Python package: datasets, preprocessing helpers, classical-feature utilities, model definitions, training loops, CKA helpers, and representation extraction. |
| `src/models/` | Model implementations for CNN1D/CNN2D, BiLSTM, Transformer, Mamba-1, and Mamba-2. |
| `src/training/` | Shared deep-model training logic, classical-model training helpers, metrics, logging, checkpoint, and evaluation utilities. |
| `scripts/01_data/` | Dataset split resolution, label-map creation, FMA retrieval utilities, mel preprocessing, rhythm feature extraction, feature-matrix construction, and validation scripts. |
| `scripts/02_train/` | Training entrypoints for all headline model families: `train_deep.py`, `train_cnn.py`, `train_transformer.py`, `train_svm.py`, `train_random_forest.py`, and `train_xgboost.py`. |
| `scripts/03_analysis/` | Representation extraction, CKA, error analysis, time-context analysis, probe transfer, and untrained-floor analysis scripts. |
| `scripts/04_freeze/` | Result aggregation, frozen-manifest construction, paired statistical tests, canonical CKA/error analysis, paper asset generation, and stretch validation. |
| `data/` | Included, small metadata needed for core reproducibility: split files, label maps, class weights, benchmark-ready manifests, and sanitized FMA retrieval metadata. Raw audio and processed spectrogram tensors are intentionally not tracked. |
| `results/` | Compact empirical artifacts used by the paper: headline metrics, CKA outputs, paired tests, ablation tables, long-sequence pilot outputs, reliability curves, time-context outputs, and paper-ready tables. Per-run manifests, machine-local environment files, and provider-specific run dumps are intentionally excluded. |
| `figures/` | Paper and appendix figures collected into one folder, including CKA plots, confusion matrices, t-SNE projections, reliability diagrams, and time-context plots. |
| `docs/paper/` | Current technical report PDF and notes on manuscript status. |
| `requirements.txt` | Python package dependencies for training and analysis. |
| `.gitignore` | Excludes local environments, raw/processed data, checkpoints, logs, and common generated files. |

## Environment

Recommended runtime:

- Python 3.10
- CUDA-capable PyTorch environment for deep training
- `ffmpeg` available on `PATH` for audio decoding and validation
- A CUDA capable GPU with enough memory (peak allocated CUDA memory reached about ~6 GB) for the selected model; Mamba runs additionally require `mamba-ssm`, `causal-conv1d`, and `einops`

Install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For macOS with Homebrew, install the system audio dependency with:

```bash
brew install ffmpeg
```

For Linux:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

## Data Setup

For size constraint reasons, this repository does not include raw FMA audio, processed spectrogram tensors, model checkpoints, or large generated binaries. To rerun training from scratch, download FMA Medium audio and FMA metadata from the official FMA dataset release at https://github.com/mdeff/fma, then place them under:

```text
data/raw/fma_medium/
data/raw/fma_metadata/
```

Expected metadata files include the FMA `tracks.csv` and `features.csv` files. The benchmark-ready split metadata already lives under `data/splits/fma_medium/`.

A typical preprocessing path is:

```bash
python scripts/01_data/resolve_fma_medium_splits.py
python scripts/01_data/prepare_label_map.py
python scripts/01_data/preprocess_fma_medium.py
python scripts/01_data/extract_rhythm_features.py
python scripts/01_data/build_classical_features.py
python scripts/01_data/dataset_sanity_check.py
```

Use `--help` on any script to inspect optional paths, worker counts, smoke settings, and validation controls.

## Running Training

Deep models use the shared `train_deep.py` entrypoint. The headline deep model keys are:

```text
cnn2d
lstm
transformer
mamba1
mamba2
```

Example single deep run:

```bash
python scripts/02_train/train_deep.py \
  --model cnn2d \
  --seed 42 \
  --output-dir results/final/cnn2d_seed42 \
  --device cuda \
  --num-workers 4
```

Five-seed deep sweep:

```bash
for model in cnn2d lstm transformer mamba1 mamba2; do
  for seed in 42 43 44 45 46; do
    python scripts/02_train/train_deep.py \
      --model "$model" \
      --seed "$seed" \
      --output-dir "results/final/${model}_seed${seed}" \
      --device cuda \
      --num-workers 4
  done
done
```

Classical baselines:

```bash
for seed in 42 43 44 45 46; do
  python scripts/02_train/train_svm.py --seed "$seed" --output-dir "results/final/svm_seed${seed}"
  python scripts/02_train/train_random_forest.py --seed "$seed" --output-dir "results/final/random_forest_seed${seed}"
  python scripts/02_train/train_xgboost.py --seed "$seed" --output-dir "results/final/xgboost_seed${seed}"
done
```

For faster artifact checks, the classical scripts support `--smoke`, and `train_deep.py` supports `--debug-gates-only` and `--forward-smoke-only`.

## Larger Sweeps

The training and analysis scripts are designed to run on local or remote GPU machines as long as the repository, dependencies, and FMA data are already present. Provider-specific job launchers, raw run directories, and machine-local logs are not tracked in this public repository. Please treat my included summaries and paper-facing artifacts as the stable surface for reviewing the reported results.

## Rebuilding Results and Figures

The included `results/` directory already contains a compact version of artifacts used by the paper. To rebuild summary tables and figures after rerunning experiments:

```bash
python scripts/04_freeze/build_frozen_manifest.py
python scripts/04_freeze/aggregate_metrics.py
python scripts/04_freeze/paired_tests.py
python scripts/04_freeze/run_canonical_cka.py
python scripts/04_freeze/run_canonical_error_analysis.py
python scripts/04_freeze/build_paper_assets.py
```

Important outputs include:

| Artifact | Purpose |
|---|---|
| `results/aggregated/headline_metrics.json` | Headline mean/std metrics by model. |
| `results/aggregated/paired_tests.json` | Paired statistical tests across deep architectures. |
| `results/paper_assets/table1_performance.csv` | Main performance table. |
| `results/paper_assets/table2_efficiency_calibration.csv` | Deep-model calibration and top-k summary. |
| `results/paper_assets/table3_top_confusions.csv` | Confusion-pair summary. |
| `results/paper_assets/table4_ablation.csv` | 15-second crop ablation summary. |
| `results/paper_assets/table5_long_sequence.csv` | 60-second long-sequence pilot summary. |
| `results/cka/` | Inter-architecture CKA, classical-feature CKA, novelty gaps, and seed-42 representation alignment outputs. |
| `results/cka_untrained_floor_deep_dive/` | Multi-initialization untrained-floor control for the convergence analysis. |

Figures are available both under `results/paper_assets/figures/` and the collected `figures/` directory.

## Verification

For a fast sanity check of the published surface, run:

```bash
python scripts/verify_publication.py
```

This checks that the `src` package imports, runs small CPU model forward passes when PyTorch is available, and confirms the presence and shape of the paper-facing artifacts under `results/` and `docs/paper/`. It is a quick integrity check and not a substitute for full GPU retraining.

## Paper Relationship

This repository is the reproducibility surface for the study experiments, not a full archival dump. It contains:

- All source code needed to reproduce preprocessing, model training, aggregation, CKA, error analysis, ablations, and figures.
- Small split metadata and result artifacts needed to verify the paper's reported conclusions.
- The current technical report PDF under `docs/paper/`.

It intentionally omits:

- Raw FMA audio.
- Processed mel-spectrogram tensor caches.
- Model checkpoints.
- Local virtual environments, provider-specific run directories, and machine-local logs.

The paper's main claims should be checked against the frozen manifest and the compact artifacts in `results/`. Full retraining requires recreating the raw and processed data directories from FMA, then running the scripts above.

All checkpoints, spectrograms, and other large artifacts may be available on request to daniel[at]danielju[dot]com.