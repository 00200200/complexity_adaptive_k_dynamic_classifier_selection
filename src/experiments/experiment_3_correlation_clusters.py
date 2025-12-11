import os
import warnings

import numpy as np
import pandas as pd
from deslib.dcs import LCA, MLA, OLA, Rank
from deslib.des import DESMI, KNOP, KNORAE, KNORAU
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import RepeatedStratifiedKFold

from src.data.dataio import load_data
from src.utils.utils import (
    calculate_k_scores,
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


def get_best_k(X, y, min_cluster_size=100):
    best_k, best_silhouette = 1, -1.0
    best_labels = np.zeros(len(X), dtype=int)
    max_k = min(int(X.shape[0] / 100), 20)

    for k in range(2, max_k + 1):
        try:
            labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
            _, counts = np.unique(labels, return_counts=True)
            if counts.min() < min_cluster_size:
                continue

            sil = silhouette_score(X, labels)
            if sil > best_silhouette:
                best_silhouette, best_k, best_labels = sil, k, labels
        except Exception:
            continue

    return best_k, best_labels


def analyze_dataset(dataset_name: str, algorithms: dict[str, object]):
    df = load_data(dataset_name)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    min_cluster_size = 100
    best_k, best_labels = get_best_k(X, y, min_cluster_size)

    cluster_labels = best_labels if best_k > 1 else np.zeros(len(X), dtype=int)

    rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

    comp_names = get_pycol_complexity_names()
    results = []

    for cluster_id in np.unique(cluster_labels):
        print(f"Processing cluster {cluster_id}")
        mask = cluster_labels == cluster_id
        Xc, yc = X[mask], y[mask]
        if len(Xc) < min_cluster_size or len(np.unique(yc)) < 2:
            continue
        _, counts = np.unique(yc, return_counts=True)
        min_samples_per_class = 20
        if counts.min() < min_samples_per_class:
            continue

        try:
            rkf.get_n_splits(Xc, yc)
        except ValueError:
            continue

        cluster_rows = []
        cluster_valid = True

        for split_idx, (train_idx, test_idx) in enumerate(rkf.split(Xc, yc), start=1):
            X_train_full, X_test = Xc[train_idx], Xc[test_idx]
            y_train_full, y_test = yc[train_idx], yc[test_idx]

            try:
                X_train, X_dsel, y_train, y_dsel, scaler = prepare_train_dsel_split(
                    X_train_full, y_train_full
                )
            except ValueError:
                continue

            if X_dsel.shape[0] < 15 or len(np.unique(y_dsel)) < 2:
                cluster_valid = False
                break

            X_test = scaler.transform(X_test)
            comp_vec, comp_score = get_pycol_complexity_vector(X_dsel, y_dsel)
            pool = train_classifiers(X_train, y_train)

            n_samples, n_features = Xc.shape
            repeat, fold = get_repeat_fold(split_idx, n_splits=10)
            pos_ratio = (
                float((y_train_full == 1).mean())
                if len(np.unique(y_train_full)) == 2
                else np.nan
            )
            _, counts = np.unique(cluster_labels, return_counts=True)

            row = {
                "dataset": f"{dataset_name}_cluster{cluster_id + 1}",
                "repeat": repeat,
                "fold": fold,
                "n_samples": n_samples,
                "n_features": n_features,
                "n_train_full": len(X_train_full),
                "n_train": len(X_train),
                "n_dsel": len(X_dsel),
                "n_test": len(X_test),
                "pos_ratio": pos_ratio,
                "n_clusters": best_k,
                "cluster_sizes": str(counts.tolist()),
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

            cluster_rows.append(row)

        if cluster_valid:
            results.extend(cluster_rows)

    return results


def _run_one_dataset(dataset_name: str, algorithms: dict[str, object]):
    set_determinism(42)
    return analyze_dataset(dataset_name, algorithms)


def run_experiment():
    output_dir = "experiments_results/experiment_3"
    os.makedirs(output_dir, exist_ok=True)
    datasets = get_dataset_names()

    algorithms = {
        "LCA": LCA,
        "MLA": MLA,
        "DESMI": DESMI,
        "OLA": OLA,
        "KNOP": KNOP,
        "KNORAE": KNORAE,
        "KNORAU": KNORAU,
        "Rank": Rank,
    }
    all_rows_nested = Parallel(n_jobs=-1)(
        delayed(_run_one_dataset)(dataset_name, algorithms) for dataset_name in datasets
    )
    all_rows: list[dict] = [row for chunk in all_rows_nested for row in chunk]

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
