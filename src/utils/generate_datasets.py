import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.utils import get_pycol_complexity_names


def parse_unique_k_list(series):
    def _parse(s):
        s = "" if pd.isna(s) else str(s).strip()
        if s == "" or ";" in s:
            return np.nan
        try:
            return float(s)
        except Exception:
            return np.nan

    return series.apply(_parse)


def get_algorithms_from_df(df):
    k_cols = [col for col in df.columns if col.endswith("_optimal_k")]
    list_cols = [col for col in df.columns if col.endswith("_optimal_k_list")]
    alg_from_k = [col.replace("_optimal_k", "") for col in k_cols]
    alg_from_list = [col.replace("_optimal_k_list", "") for col in list_cols]
    return sorted(set(alg_from_k + alg_from_list))


def main(
    experiment_nr: str,
    csv_filename: str = "optimal_k_results.csv",
    drop_na: bool = False,
):
    input_file = Path(f"experiments_results/{experiment_nr}/{csv_filename}")
    df = pd.read_csv(input_file)

    complexity_all = get_pycol_complexity_names()
    complexity_cols = [c for c in complexity_all if c in df.columns]
    algorithms = get_algorithms_from_df(df)

    output_dir = Path(f"data/generated_datasets/{experiment_nr}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for alg in algorithms:
        list_col = f"{alg}_optimal_k_list"
        if list_col not in df.columns:
            print(f"Skipping {alg}: no {list_col} column found")
            continue

        y_series = parse_unique_k_list(df[list_col])
        y_col = f"{alg}_optimal_k"
        X = df[complexity_cols].copy()
        y = y_series.copy()

        print(f"\n{alg}: {(~y.isna()).sum()} valid rows")

        nan_metrics = [
            (col, X[col].isna().sum())
            for col in complexity_cols
            if X[col].isna().sum() > 0
        ]
        if nan_metrics:
            print("  Metrics with NaN:")
            for metric, count in sorted(nan_metrics, key=lambda x: x[1], reverse=True)[
                :5
            ]:
                print(f"    - {metric}: {count} ({count / len(X) * 100:.1f}%)")

        mask = pd.Series(True, index=X.index)
        if drop_na:
            mask &= ~(X.isna().any(axis=1) | y.isna())

        X_clean, y_clean = X[mask], y[mask]

        dataset_col = None
        if "dataset" in df.columns:
            dataset_col = (
                df.loc[mask, "dataset"]
                .copy()
                .astype(str)
                .str.replace(r"_cluster\d*$", "", regex=True)
            )

        parts = [X_clean, y_clean.rename(y_col)]
        if dataset_col is not None:
            parts.append(dataset_col.rename("dataset"))

        dataset = pd.concat(parts, axis=1).drop_duplicates()
        output_file = output_dir / f"{alg}.csv"
        dataset.to_csv(output_file, index=False)
        print(f"  → Saved: {len(dataset)} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate datasets from experiment results"
    )
    parser.add_argument("experiment", help="Experiment number (e.g., experiment_4)")
    parser.add_argument(
        "--csv",
        default="optimal_k_results.csv",
        help="CSV filename inside the experiment folder (e.g., optimal_k_results.csv)",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop rows with NaN in complexity metrics (default: keep rows with NaN)",
    )

    args = parser.parse_args()
    main(args.experiment, csv_filename=args.csv, drop_na=args.drop)
