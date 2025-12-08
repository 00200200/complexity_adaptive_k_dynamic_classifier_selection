import os
import warnings

import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold

from src.data.dataio import load_data
from src.utils.utils import (
    calculate_k_scores,
    create_boxplot,
    get_algorithms,
    get_best_ks_from_scores,
    get_dataset_names,
    get_repeat_fold,
    prepare_train_dsel_split,
    set_determinism,
    train_classifiers,
)

warnings.filterwarnings("ignore")


def analyze_dataset(dataset_name, algorithms):
    df = load_data(dataset_name)
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

    n_splits, n_repeats = 5, 2
    rkf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42
    )
    total_splits = rkf.get_n_splits(X, y)

    results = []
    for split_idx, (train_idx, test_idx) in enumerate(rkf.split(X, y), start=1):
        X_train_full, X_test = X[train_idx], X[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]

        X_train, X_dsel, y_train, y_dsel, scaler = prepare_train_dsel_split(
            X_train_full, y_train_full
        )
        X_test = scaler.transform(X_test)
        pool = train_classifiers(X_train, y_train)

        repeat, fold = get_repeat_fold(split_idx, n_splits)
        row = {"dataset": dataset_name, "repeat": repeat, "fold": fold}

        for alg_name, alg_class in algorithms.items():
            k_scores = calculate_k_scores(
                alg_class, pool, X_dsel, y_dsel, X_test, y_test
            )
            best_ks = get_best_ks_from_scores(k_scores)
            if best_ks:
                row[f"{alg_name}_optimal_k_list"] = ";".join(map(str, best_ks))
            for k, score in k_scores.items():
                row[f"{alg_name}_k_{k}"] = score

        results.append(row)

        if split_idx % 5 == 0 or split_idx == total_splits:
            print(
                f"  [{dataset_name}] Split {split_idx}/{total_splits} done, {total_splits - split_idx} remaining"
            )
    return results


def _run_one_dataset(dataset_name, algorithms):
    set_determinism(42)
    return analyze_dataset(dataset_name, algorithms)


def run_experiment():
    output_dir = "experiments_results/experiment_1"
    os.makedirs(output_dir, exist_ok=True)

    datasets = get_dataset_names()
    algorithms = get_algorithms()

    all_rows_nested = Parallel(n_jobs=-1)(
        delayed(_run_one_dataset)(name, algorithms) for name in datasets
    )
    all_rows = [row for chunk in all_rows_nested for row in chunk]

    if not all_rows:
        print("No results to analyze")
        return

    df = (
        pd.DataFrame(all_rows)
        .sort_values(["dataset", "repeat", "fold"])
        .reset_index(drop=True)
    )
    df.to_csv(f"{output_dir}/optimal_k_results.csv", index=False)

    def parse_k_list(k_list_str):
        if pd.isna(k_list_str):
            return []
        try:
            return [int(k.strip()) for k in str(k_list_str).split(";") if k.strip()]
        except Exception:
            return []

    all_k_data = []
    for _, row in df.iterrows():
        for col in [c for c in df.columns if c.endswith("_optimal_k_list")]:
            alg = col.replace("_optimal_k_list", "")
            for k in parse_k_list(row[col]):
                all_k_data.append({"dataset": row["dataset"], "Algorithm": alg, "K": k})

    k_df = pd.DataFrame(all_k_data)
    mode_data = (
        k_df.groupby(["dataset", "Algorithm"])["K"]
        .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.min())
        .reset_index()
        .rename(columns={"K": "mode_k"})
    )

    mean_data = (
        k_df.groupby(["dataset", "Algorithm"])["K"]
        .mean()
        .reset_index()
        .rename(columns={"K": "mean_k"})
    )

    summary_mode = mode_data.pivot(
        index="dataset", columns="Algorithm", values="mode_k"
    ).reset_index()
    summary_mean = mean_data.pivot(
        index="dataset", columns="Algorithm", values="mean_k"
    ).reset_index()

    summary_mode.to_csv(f"{output_dir}/summary_mode.csv", index=False)
    summary_mean.to_csv(f"{output_dir}/summary_mean.csv", index=False)

    for col in summary_mode.columns:
        if col != "dataset":
            summary_mode.rename(columns={col: f"{col}_optimal_k"}, inplace=True)
            summary_mean.rename(columns={col: f"{col}_optimal_k"}, inplace=True)

    create_boxplot(
        summary_mode, save_path=f"{output_dir}/optimal_k_distribution_mode.eps"
    )
    create_boxplot(
        summary_mean, save_path=f"{output_dir}/optimal_k_distribution_average.eps"
    )


if __name__ == "__main__":
    run_experiment()
