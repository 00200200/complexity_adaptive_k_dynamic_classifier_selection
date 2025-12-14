import argparse
import os

import numpy as np
import pandas as pd

from src.utils.utils import (
    create_correlation_heatmap,
    create_individual_correlation_plots,
    get_pycol_complexity_names,
)


def parse_optimal_k_columns(df):
    alg_ok_list_cols = [c for c in df.columns if c.endswith("_optimal_k_list")]

    for col in alg_ok_list_cols:
        alg = col.replace("_optimal_k_list", "")
        opt_col = f"{alg}_optimal_k"
        lists = df[col].fillna("").astype(str)

        if opt_col not in df.columns:
            df[opt_col] = np.nan

        is_single = ~lists.str.contains(";") & (lists != "")
        df.loc[~is_single, opt_col] = np.nan
        df.loc[is_single, opt_col] = pd.to_numeric(lists[is_single], errors="coerce")


def get_group_column(df):
    if "dataset_clusters" in df.columns:
        return "dataset_clusters"

    df["dataset_group"] = (
        df["dataset"].astype(str).str.replace(r"_cluster\d+$", "_clusters", regex=True)
    )
    return "dataset_group"


def main(experiment_nr, csv_filename="optimal_k_results.csv", drop_na=False):
    csv_path = os.path.abspath(f"experiments_results/{experiment_nr}/{csv_filename}")
    out_dir = os.path.abspath(f"experiments_results/{experiment_nr}/plots")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "dataset" not in df.columns:
        raise ValueError("CSV must contain 'dataset' column")

    pycol_metrics = get_pycol_complexity_names()
    parse_optimal_k_columns(df)

    corr_df = create_correlation_heatmap(
        df, out_dir, complexity_names=pycol_metrics, drop_na=drop_na
    )

    if corr_df is not None and not corr_df.empty:
        group_col = get_group_column(df)
        create_individual_correlation_plots(
            corr_df,
            df,
            out_dir,
            group_col=group_col,
            complexity_names=pycol_metrics,
            drop_na=drop_na,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots from pycol-complexity experiment results"
    )
    parser.add_argument("experiment", help="Experiment number (e.g., experiment_2)")
    parser.add_argument(
        "--csv",
        default="optimal_k_results.csv",
        help="CSV filename inside the experiment folder",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Use listwise deletion - drop rows with NaN in any complexity metric (default: pairwise deletion)",
    )
    args = parser.parse_args()
    main(args.experiment, csv_filename=args.csv, drop_na=args.drop)
