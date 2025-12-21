# Complexity-Adaptive k for Dynamic Classifier Selection

Research implementation exploring the relationship between dataset complexity and optimal k parameter in Dynamic Classifier Selection methods.

## Overview

This project investigates how dataset complexity metrics can be used to adaptively determine the k parameter in DCS/DES techniques, potentially improving classification performance on heterogeneous datasets.

## Project Structure

- `src/models/` - Custom implementations (PyColKNN, PyColCluster)
- `src/experiments/` - Five experimental scenarios
- `src/data/` - Data loading utilities
- `src/utils/` - Analysis and plotting tools
- `generated_datasets/` - Experimental results

## Experiments

1. Optimal k determination for individual datasets
2. Correlation analysis between optimal k and complexity metrics
3. Cluster-based complexity analysis
4. Meta-classifier approach for k prediction
5. Adaptive vs. baseline DCS comparison

## Requirements

Python 3.11+ and dependencies listed in `pyproject.toml`. Key libraries: scikit-learn, DESlib, pandas, numpy.

## Installation

```bash
uv sync
```

## Usage

Run individual experiments from the `src/experiments/` directory. Each script is self-contained and generates results in the `generated_datasets/` or `eperiment/results` folder.
