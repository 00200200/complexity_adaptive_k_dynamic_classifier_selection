import argparse
import os
import warnings

import numpy as np
import pandas as pd
from deslib.des import KNORAE
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.data.dataio import load_data
from src.models.PyColCluster import AdaptiveClusterPycol
from src.models.PyColKNN import AdaptivePyColKNN
from src.utils.utils import get_dataset_names, set_determinism, train_classifiers

warnings.filterwarnings("ignore")

EXP_SOURCE = "experiment_3"

THRESHOLD = 0.85

ALGORITHM_CONFIG_EXP3 = {
    "KNORAE": {
        "algorithm": KNORAE,
        "features": ["Purity", "D3_value", "LSC"],
        "meta_model": GaussianNB(),
        "sampling": "smote",
        "threshold": 0.85,
    },
}


def get_optimal_k(dataset_name, algorithm_name, optimal_k_df):
    try:
        k = int(
            optimal_k_df[optimal_k_df["dataset"] == dataset_name][algorithm_name].iloc[
                0
            ]
        )
        return k if 3 <= k <= 15 else 7
    except Exception:
        return 7


def run_fold(
    dataset_name, algorithm_name, fold_num, train_idx, test_idx, X, y, optimal_k_df
):
    if EXP_SOURCE == "experiment_3":
        config = ALGORITHM_CONFIG_EXP3[algorithm_name]
    else:
        return None

    X_train_full, X_test = X[train_idx], X[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]

    X_train, X_dsel, y_train, y_dsel = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=42,
        stratify=y_train_full,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dsel, X_test = scaler.transform(X_dsel), scaler.transform(X_test)

    pool = train_classifiers(X_train, y_train)
    base_k = get_optimal_k(dataset_name, algorithm_name, optimal_k_df)

    base = config["algorithm"](pool_classifiers=pool, k=base_k, random_state=42)
    base.fit(X_dsel, y_dsel)
    base_bacc = balanced_accuracy_score(y_test, base.predict(X_test))

    meta_df = pd.read_csv(f"data/generated_datasets/{EXP_SOURCE}/{algorithm_name}.csv")
    meta_df = meta_df[meta_df["dataset"] != dataset_name]

    X_meta_raw = meta_df[config["features"]].values
    y_meta = meta_df[f"{algorithm_name}_optimal_k"].values

    mask = ~(np.isnan(X_meta_raw).any(axis=1) | np.isnan(y_meta))
    X_meta_raw = X_meta_raw[mask]
    y_meta = y_meta[mask]

    complexity_scaler = StandardScaler()
    X_meta = complexity_scaler.fit_transform(X_meta_raw)

    unique_classes = np.unique(y_meta)
    class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
    inverse_mapping = {idx: cls for cls, idx in class_mapping.items()}
    y_meta_mapped = np.array([class_mapping[val] for val in y_meta])

    if config["sampling"] == "undersample":
        try:
            X_meta, y_meta_mapped = RandomUnderSampler(random_state=42).fit_resample(
                X_meta, y_meta_mapped
            )
        except Exception:
            pass
    elif config["sampling"] == "smote":
        try:
            X_meta, y_meta_mapped = SMOTE(random_state=42).fit_resample(
                X_meta, y_meta_mapped
            )
        except Exception:
            pass

    meta_model = clone(config["meta_model"])
    if config["sampling"] == "balancing":
        sample_weights = compute_sample_weight("balanced", y_meta_mapped)
        meta_model.fit(X_meta, y_meta_mapped, sample_weight=sample_weights)
    else:
        meta_model.fit(X_meta, y_meta_mapped)

    adaptive_cluster = AdaptiveClusterPycol(
        pool_classifiers=pool,
        algorithm=config["algorithm"],
        meta_model=meta_model,
        threshold=THRESHOLD,
        fallback_k=base_k,
        selected_feature_names=config["features"],
        random_state=42,
        complexity_scaler=complexity_scaler,
        k_mapping=inverse_mapping,
    )
    adaptive_cluster.fit(X_dsel, y_dsel)
    adaptive_cluster_bacc = balanced_accuracy_score(
        y_test, adaptive_cluster.predict(X_test)
    )

    adaptive_pycol_knn = AdaptivePyColKNN(
        pool_classifiers=pool,
        algorithm=config["algorithm"],
        meta_model=meta_model,
        threshold=THRESHOLD,
        fallback_k=base_k,
        selected_feature_names=config["features"],
        random_state=42,
        complexity_scaler=complexity_scaler,
        k_mapping=inverse_mapping,
    )
    adaptive_pycol_knn.fit(X_dsel, y_dsel)
    adaptive_pycol_knn_bacc = balanced_accuracy_score(
        y_test, adaptive_pycol_knn.predict(X_test)
    )

    return {
        "algorithm": algorithm_name,
        "dataset": dataset_name,
        "fold": fold_num,
        "base_k": base_k,
        "base_bacc": base_bacc,
        "adaptive_cluster_bacc": adaptive_cluster_bacc,
        "adaptive_pycol_knn_bacc": adaptive_pycol_knn_bacc,
        "delta_cluster": adaptive_cluster_bacc - base_bacc,
        "delta_pycol_knn": adaptive_pycol_knn_bacc - base_bacc,
    }


def _run_single_fold(
    dataset_name, algorithm_name, fold_num, train_idx, test_idx, X, y, optimal_k_df
):
    set_determinism(42 + fold_num)
    try:
        return run_fold(
            dataset_name,
            algorithm_name,
            fold_num,
            train_idx,
            test_idx,
            X,
            y,
            optimal_k_df,
        )
    except Exception:
        return None


def prepare_fold_tasks(algorithm_name, optimal_k_df):
    tasks = []
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    for dataset_name in get_dataset_names():
        try:
            df = load_data(dataset_name)
            X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
            for fold_num, (train_idx, test_idx) in enumerate(rkf.split(X, y), 1):
                tasks.append(
                    (
                        dataset_name,
                        algorithm_name,
                        fold_num,
                        train_idx,
                        test_idx,
                        X,
                        y,
                        optimal_k_df,
                    )
                )
        except Exception:
            pass

    return tasks


def main(algorithms=None):
    if EXP_SOURCE == "experiment_3":
        ALGORITHM_CONFIG = ALGORITHM_CONFIG_EXP3
    else:
        return

    optimal_k_df = pd.read_csv("experiments_results/experiment_1/summary_mode.csv")
    algorithms = algorithms or list(ALGORITHM_CONFIG.keys())

    all_results = []

    for alg in algorithms:
        if alg not in ALGORITHM_CONFIG:
            continue

        data_path = f"data/generated_datasets/{EXP_SOURCE}/{alg}.csv"
        if not os.path.exists(data_path):
            continue

        tasks = prepare_fold_tasks(alg, optimal_k_df)

        results = Parallel(n_jobs=-1, verbose=0)(
            delayed(_run_single_fold)(*task) for task in tasks
        )
        alg_results = [r for r in results if r is not None]
        all_results.extend(alg_results)

    if not all_results:
        return

    df = pd.DataFrame(all_results)
    output_dir = f"experiments_results/experiment_5/{EXP_SOURCE}/THRESHOLD_{THRESHOLD}"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/results_all_algorithms.csv", index=False)

    for alg in algorithms:
        df_alg = df[df["algorithm"] == alg]
        if not df_alg.empty:
            df_alg.to_csv(f"{output_dir}/results_{alg}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 5: Adaptive vs Base DCS")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=list(ALGORITHM_CONFIG_EXP3.keys()),
        help="Algorytmy do przetestowania (domyślnie: wszystkie)",
    )

    args = parser.parse_args()
    main(algorithms=args.algorithms)
