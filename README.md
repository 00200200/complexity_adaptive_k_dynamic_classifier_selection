# Complexity-Adaptive k for Dynamic Classifier Selection

Research implementation exploring the relationship between dataset complexity and the neighborhood size `k` in Dynamic Classifier Selection (DCS) and Dynamic Ensemble Selection (DES).

## Overview

The code measures data complexity, trains meta-classifiers to predict `k`, and compares adaptive selection with fixed-k baselines. Two custom wrappers estimate complexity around a query point or within a cluster, and fall back to a baseline `k` when the prediction is unavailable or below a confidence threshold. Whether this improves classification is an experimental question; datasets and result tables are not included in this repository.

## Project Structure

- `src/models/` - Custom wrappers (`AdaptivePyColKNN`, `AdaptiveClusterPycol`)
- `src/experiments/` - Five experimental scenarios
- `src/data/` - Data loading utilities
- `src/utils/` - Analysis and plotting tools
- `data/` - Input CSV datasets (not included)
- `data/generated_datasets/` - Meta-classifier datasets generated from experiment outputs
- `experiments_results/` - Generated result tables and plots

## Experiments

1. Search for the best observed `k` for individual datasets
2. Correlation analysis between optimal k and complexity metrics
3. Cluster-based complexity analysis
4. Meta-classifier approach for k prediction
5. Adaptive vs. baseline DCS comparison

The search evaluates odd `k` values from 3 to 15, subject to the available selection-set size, using balanced accuracy. The scores used to choose `k` are not an independent evaluation of the selected value. Experiment 5 currently configures KNORAE and compares a fixed-k baseline with both adaptive wrappers.

## Requirements

Python 3.11+, the `uv` package manager, and dependencies listed in `pyproject.toml`. Key libraries include scikit-learn, DESlib, pycol-complexity, pandas, and NumPy.

## Installation

```bash
uv sync
```

## Usage

Run modules from the repository root so imports and relative data paths resolve correctly. First provide `data/<dataset_name>.csv` files for the names in `get_dataset_names()` in `src/utils/utils.py`, or edit that list for your datasets. CSV files need a header row, numeric feature columns, and the target label in the last column; no dataset downloader is included.

```bash
# Experiment 1: k search and baseline summary
uv run python -m src.experiments.experiment_1_optimal_k_for_dataset

# Experiment 2: dataset-level complexity and k results
uv run python -m src.experiments.experiment_2_correlation_optimal_k

# Experiment 3: cluster-level complexity and k results
uv run python -m src.experiments.experiment_3_correlation_clusters

# Prepare the meta-classifier data used by the current experiments 4 and 5
uv run python -m src.utils.generate_datasets experiment_3

# Experiment 4: compare meta-classifiers
uv run python -m src.experiments.experiment_4_metaclassifier

# Experiment 5: compare adaptive selection with the baseline
uv run python -m src.experiments.experiment_5_adaptive_vs_base --algorithms KNORAE
```

Experiments 4 and 5 default to `data/generated_datasets/experiment_3/`. Experiment 5 also reads `experiments_results/experiment_1/summary_mode.csv`; it trains its own meta-classifier rather than loading a model from experiment 4. These scripts are therefore not independent entry points. Results are written under `experiments_results/experiment_<n>/`.

The experiments use repeated cross-validation, and several parallelize across all available CPU cores. Review the dataset list and script settings before starting a run.
