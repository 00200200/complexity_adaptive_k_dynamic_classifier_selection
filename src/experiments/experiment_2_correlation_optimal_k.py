import os
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold

from src.data.dataio import load_data
from src.utils.utils import (
    calculate_k_scores,
    get_algorithms,
    get_best_ks_from_scores,
    get_dataset_names,
    get_pycol_complexity_names,
    get_pycol_complexity_vector,
    get_repeat_fold,
    prepare_train_dsel_split,
    set_determinism,
    train_classifiers,
)

warnings.filterwarnings("ignore")


def analyze_dataset(dataset_name, algorithms):
    df = load_data(dataset_name)
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    n_samples, n_features = X.shape

    n_splits, n_repeats = 10, 10
    rkf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42
    )
    comp_names = get_pycol_complexity_names()
    results = []

    for split_idx, (train_idx, test_idx) in enumerate(rkf.split(X, y), start=1):
        try:
            X_train_full, X_test = X[train_idx], X[test_idx]
            y_train_full, y_test = y[train_idx], y[test_idx]

            X_train, X_dsel, y_train, y_dsel, scaler = prepare_train_dsel_split(
                X_train_full, y_train_full
            )
            X_test = scaler.transform(X_test)

            comp_vec, comp_score = get_pycol_complexity_vector(X_dsel, y_dsel)
            pool = train_classifiers(X_train, y_train)

            repeat, fold = get_repeat_fold(split_idx, n_splits)
            pos_ratio = (
                float((y_train_full == 1).mean())
                if len(np.unique(y_train_full)) == 2
                else np.nan
            )

            row = {
                "dataset": dataset_name,
                "repeat": repeat,
                "fold": fold,
                "n_samples": n_samples,
                "n_features": n_features,
                "n_train_full": len(X_train_full),
                "n_train": len(X_train),
                "n_dsel": len(X_dsel),
                "n_test": len(X_test),
                "pos_ratio": pos_ratio,
                "complexity_score": comp_score,
            }
            row.update({name: float(val) for name, val in zip(comp_names, comp_vec)})

            for alg_name, alg_class in algorithms.items():
                k_scores = calculate_k_scores(
                    alg_class, pool, X_dsel, y_dsel, X_test, y_test
                )
                best_ks = get_best_ks_from_scores(k_scores)
                row[f"{alg_name}_optimal_k_list"] = (
                    ";".join(map(str, best_ks)) if best_ks else ""
                )
                for k, score in k_scores.items():
                    row[f"{alg_name}_k_{k}"] = score

            results.append(row)
        except Exception:
            continue

    return results


def _run_one_dataset(dataset_name, algorithms):
    set_determinism(42)
    return analyze_dataset(dataset_name, algorithms)


def run_experiment():
    output_dir = "experiments_results/experiment_2"
    os.makedirs(output_dir, exist_ok=True)

    all_rows_nested = Parallel(n_jobs=-1)(
        delayed(_run_one_dataset)(name, get_algorithms())
        for name in get_dataset_names()
    )
    all_rows = [row for chunk in all_rows_nested for row in chunk]

    if not all_rows:
        return

    df = (
        pd.DataFrame(all_rows)
        .sort_values(["dataset", "repeat", "fold"])
        .reset_index(drop=True)
    )
    df.to_csv(f"{output_dir}/optimal_k_results.csv", index=False)


if __name__ == "__main__":
    run_experiment()
