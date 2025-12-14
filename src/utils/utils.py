import os
import random
import tempfile
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from deslib.dcs import LCA, MCB, MLA, OLA, APosteriori, APriori, Rank
from deslib.des import DESKNN, DESMI, KNOP, KNORAE, KNORAU, METADES
from scipy.stats import spearmanr
from sklearn.ensemble import (
    BaggingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


def calculate_k_scores(algorithm_class, pool, X_dsel, y_dsel, X_test, y_test):
    if X_dsel.shape[0] < 6 or len(np.unique(y_dsel)) < 2 or len(np.unique(y_test)) < 2:
        return {}

    max_k = min(15, len(X_dsel) - 1)
    ks = list(range(3, max_k + 1, 2)) if max_k >= 3 else []

    scores = {}
    for k in ks:
        try:
            model = algorithm_class(pool_classifiers=pool, k=k)
            model.fit(X_dsel, y_dsel)
            scores[k] = balanced_accuracy_score(y_test, model.predict(X_test))
        except Exception:
            continue
    return scores


def get_algorithms():
    return {
        "LCA": LCA,
        "MLA": MLA,
        "DESMI": DESMI,
        "OLA": OLA,
        "KNOP": KNOP,
        "KNORAE": KNORAE,
        "KNORAU": KNORAU,
        "Rank": Rank,
        "DESKNN": DESKNN,
        "METADES": METADES,
        "MCB": MCB,
        "APriori": APriori,
        "APosteriori": APosteriori,
    }


def get_dataset_names():
    return [
        "ring",
        "phoneme",
        "banana",
        "spambase",
        "titanic",
        "german",
        "tic-tac-toe",
        "mammographic",
        "australian",
        "crx",
        "wdbc",
        "saheart",
        "monk-2",
        "bands",
        "bupa",
        "breast",
        "heart",
        "spectfheart",
        "housevotes",
        "sonar",
    ]


def train_classifiers(X_train, y_train, random_state=42):
    classifiers = []

    for depth in [3, 5, 7, None]:
        classifiers.append(
            DecisionTreeClassifier(max_depth=depth, random_state=random_state).fit(
                X_train, y_train
            )
        )

    classifiers.append(
        RandomForestClassifier(n_estimators=100, random_state=random_state).fit(
            X_train, y_train
        )
    )
    classifiers.append(
        ExtraTreesClassifier(n_estimators=100, random_state=random_state).fit(
            X_train, y_train
        )
    )
    classifiers.append(
        BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=5, random_state=random_state),
            n_estimators=50,
            random_state=random_state,
        ).fit(X_train, y_train)
    )

    for k in [3, 5, 7, 9]:
        classifiers.append(KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train))

    for kernel in ["linear", "rbf"]:
        for C_val in [0.5, 1, 2]:
            classifiers.append(
                SVC(
                    kernel=kernel, C=C_val, probability=True, random_state=random_state
                ).fit(X_train, y_train)
            )

    for C_val in [0.5, 1, 2]:
        classifiers.append(
            LogisticRegression(C=C_val, random_state=random_state).fit(X_train, y_train)
        )

    classifiers.append(GaussianNB().fit(X_train, y_train))

    for hidden_layers in [(50,), (100,), (50, 50), (100, 50)]:
        classifiers.append(
            MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                random_state=random_state,
                early_stopping=True,
            ).fit(X_train, y_train)
        )

    return classifiers


def get_dataset_colors(df, group_col: str = "dataset"):
    groups = df[group_col].unique()
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(groups)))
    return dict(zip(groups, colors))


def setup_plot(xlabel, ylabel, figsize=(10, 8)):
    plt.figure(figsize=figsize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def save_plot(filepath):
    plt.tight_layout()
    plt.savefig(filepath, format="eps")
    png_path = (
        filepath.replace(".eps", ".png")
        if filepath.endswith(".eps")
        else f"{filepath.rsplit('.', 1)[0]}.png"
    )
    plt.savefig(png_path, format="png")
    plt.close()


def set_determinism(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def prepare_train_dsel_split(X_train_full, y_train_full):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_train, X_dsel, y_train, y_dsel = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=42,
        stratify=y_train_full,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dsel = scaler.transform(X_dsel)

    return X_train, X_dsel, y_train, y_dsel, scaler


def get_repeat_fold(split_idx, n_splits=10):
    repeat = (split_idx - 1) // n_splits + 1
    fold = (split_idx - 1) % n_splits + 1
    return repeat, fold


def get_best_ks_from_scores(k_scores):
    if not k_scores:
        return []
    max_val = max(k_scores.values())
    return sorted(k for k, v in k_scores.items() if v == max_val)


def create_boxplot(df, output_dir=None, save_path=None):
    k_cols = [c for c in df.columns if c.endswith("_optimal_k")]
    k_df = df[["dataset"] + k_cols].melt(
        id_vars="dataset", var_name="Algorithm", value_name="Optimal_K"
    )
    k_df["Algorithm"] = k_df["Algorithm"].str.replace("_optimal_k", "", regex=False)
    colors = get_dataset_colors(df)

    setup_plot("Algorithms", "Optimal K", (14, 8))
    sns.boxplot(data=k_df, x="Algorithm", y="Optimal_K", color="white", fliersize=0)
    plt.yticks([3, 5, 7, 9, 11, 13, 15])

    alg_names = sorted(k_df["Algorithm"].unique())
    alg_to_x = {alg: i for i, alg in enumerate(alg_names)}
    for dataset in df["dataset"].unique():
        data = k_df[k_df["dataset"] == dataset]
        if not len(data):
            continue
        x = [alg_to_x[alg] + np.random.uniform(-0.2, 0.2) for alg in data["Algorithm"]]
        plt.scatter(x, data["Optimal_K"], color=colors[dataset], alpha=0.6, s=30)

    plt.xticks(rotation=45, ha="right")

    path = (
        save_path
        if save_path
        else f"{output_dir}/optimal_k_distribution.eps"
        if output_dir
        else None
    )
    if not path:
        raise ValueError("No output path specified")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_plot(path)
    return k_df


def create_correlation_heatmap(
    df, output_dir, alpha=0.05, complexity_names=None, drop_na=False, drop_inf=False
):
    complexity_cols = (
        complexity_names if complexity_names else get_pycol_complexity_names()
    )
    k_cols = [col for col in df.columns if col.endswith("_optimal_k")]
    algorithms = [k_col.replace("_optimal_k", "") for k_col in k_cols]
    complexity_present = [c for c in complexity_cols if c in df.columns]

    print("Heatmap sample counts per algorithm:")
    for k_col, alg in zip(k_cols, algorithms):
        if drop_na:
            mask = (~df[k_col].isna()) & (~df[complexity_present].isna().any(axis=1))
            n_k = int(mask.sum())
        else:
            n_k = df[k_col].dropna().shape[0]
        print(f"   - {alg}: N={n_k}")

    corr_matrix = pd.DataFrame(index=algorithms, columns=complexity_cols)
    pval_matrix = pd.DataFrame(index=algorithms, columns=complexity_cols)
    correlations = []

    for k_col, alg in zip(k_cols, algorithms):
        listwise_mask = pd.Series(True, index=df.index)

        if drop_na:
            listwise_mask = listwise_mask & (~df[k_col].isna())
            listwise_mask = listwise_mask & (~df[complexity_present].isna().any(axis=1))
        if drop_inf:
            listwise_mask = listwise_mask & ~(df[k_col].isin([np.inf, -np.inf]))
            for comp_col in complexity_present:
                if comp_col in df.columns:
                    listwise_mask = listwise_mask & ~(
                        df[comp_col].isin([np.inf, -np.inf])
                    )

        for comp_col in complexity_cols:
            if comp_col in df.columns:
                try:
                    pair = (
                        df.loc[listwise_mask, [k_col, comp_col]].dropna()
                        if drop_na
                        else df[[k_col, comp_col]].dropna()
                    )
                    n_valid = len(pair)
                    corr, pval = (
                        spearmanr(pair[k_col], pair[comp_col])
                        if n_valid >= 3
                        else (np.nan, np.nan)
                    )
                    corr = 0.0 if np.isnan(corr) else float(corr)
                    pval = 1.0 if np.isnan(pval) else float(pval)
                except Exception:
                    n_valid, corr, pval = 0, 0.0, 1.0
                corr_matrix.loc[alg, comp_col] = corr
                pval_matrix.loc[alg, comp_col] = pval
                correlations.append(
                    {
                        "Algorithm": alg,
                        "Metric": comp_col,
                        "Correlation": corr,
                        "PValue": pval,
                        "N": n_valid,
                    }
                )

    plt.figure(figsize=(20, 10))
    sns.heatmap(corr_matrix.fillna(0), annot=True, cmap="RdBu_r", center=0)
    plt.title("Correlation: Complexity Metrics vs Optimal K")
    plt.xlabel("Complexity Metrics")
    plt.ylabel("Algorithms")
    plt.xticks(rotation=45)
    save_plot(f"{output_dir}/complexity_vs_k_correlation_heatmap.eps")

    plt.figure(figsize=(20, 10))
    sns.heatmap(pval_matrix.astype(float), annot=True, cmap="viridis_r", vmin=0, vmax=1)
    plt.title("Spearman p-values: Complexity vs Optimal K")
    plt.xlabel("Complexity Metrics")
    plt.ylabel("Algorithms")
    plt.xticks(rotation=45)
    save_plot(f"{output_dir}/complexity_vs_k_pvalues_heatmap.eps")

    return pd.DataFrame(correlations)


def create_individual_correlation_plots(
    corr_df,
    df,
    output_dir,
    threshold=0.2,
    alpha=0.05,
    group_col="dataset",
    show_legend=False,
    complexity_names=None,
    drop_na=False,
    drop_inf=False,
):
    if corr_df.empty or "Correlation" not in corr_df.columns:
        print("Warning: correlation data is empty, skipping individual plots")
        return False

    high_corr = corr_df[
        (abs(corr_df["Correlation"]) >= threshold) & (corr_df["PValue"] <= alpha)
    ]
    print(f"Creating {len(high_corr)} correlation plots")
    for _, r in high_corr.iterrows():
        try:
            print(
                f"   - {r['Algorithm']} vs {r['Metric']}: N={int(r['N'])}, corr={float(r['Correlation']):.3f}"
            )
        except Exception:
            pass

    group_col = group_col if group_col in df.columns else "dataset"
    colors = get_dataset_colors(df, group_col=group_col)

    for _, row in high_corr.iterrows():
        alg, metric, corr = row["Algorithm"], row["Metric"], row["Correlation"]
        if metric not in df.columns:
            continue

        os.makedirs(f"{output_dir}/metric_alg_correlations/{alg}", exist_ok=True)
        x, y = df[metric], df[f"{alg}_optimal_k"]

        list_mask = pd.Series(True, index=df.index)

        if drop_na and complexity_names:
            present = [c for c in complexity_names if c in df.columns]
            list_mask = list_mask & (~y.isna()) & (~df[present].isna().any(axis=1))

        if drop_inf:
            list_mask = list_mask & ~(x.isin([np.inf, -np.inf]))
            list_mask = list_mask & ~(y.isin([np.inf, -np.inf]))

        valid = list_mask & (~x.isna()) & (~y.isna())

        plt.figure(figsize=(10, 8))
        plt.xlabel(metric)
        plt.ylabel("Optimal K")
        plt.title(f"{alg} vs {metric}\nCorr: {corr:.3f}")

        for group_value in df[group_col].unique():
            mask = (df[group_col] == group_value) & valid
            if mask.any():
                plt.scatter(
                    x[mask], y[mask], color=colors[group_value], label=group_value
                )

        y_min, y_max = plt.ylim()
        plt.yticks(np.arange(np.ceil(y_min), np.floor(y_max) + 1)[1::2])
        if show_legend:
            plt.legend(title=group_col)
        save_plot(f"{output_dir}/metric_alg_correlations/{alg}/{metric}.eps")

    return True


from pycol_complexity import complexity


def get_pycol_complexity_names():
    return [
        "F1",
        "F1v",
        "F2",
        "F3",
        "F4",
        "input_noise",
        "R_value",
        "degOver",
        "N3",
        "SI",
        "N4",
        "kDN",
        "D3_value",
        "CM",
        "borderline",
        "N1",
        "T1",
        "Clst",
        "ONB",
        "LSC",
        "DBC",
        "N2",
        "NSG",
        "ICSV",
        "MRCA",
        "C1",
        "C2",
        "Purity",
        "neighbourhood_separability",
    ]


def get_pycol_complexity_vector(X_local, y_local, selected_features=None):
    features = selected_features or get_pycol_complexity_names()

    if len(np.unique(y_local)) < 2 or X_local.shape[0] < 2:
        return [np.nan] * len(features), np.nan

    X_local = X_local.reshape(1, -1) if X_local.ndim == 1 else X_local

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        pd.DataFrame(X_local).assign(target=y_local).to_csv(tmp.name, index=False)
        temp_path = tmp.name

    try:
        calc = complexity.Complexity(
            temp_path, distance_func="default", file_type="csv"
        )

        measure_map = {
            "F1": calc.F1,
            "F1v": calc.F1v,
            "F2": calc.F2,
            "F3": calc.F3,
            "F4": calc.F4,
            "input_noise": calc.input_noise,
            "R_value": calc.R_value,
            "degOver": calc.deg_overlap,
            "N3": calc.N3,
            "SI": calc.SI,
            "N4": calc.N4,
            "kDN": calc.kDN,
            "D3_value": calc.D3_value,
            "CM": calc.CM,
            "borderline": calc.borderline,
            "N1": calc.N1,
            "T1": calc.T1,
            "Clst": calc.Clust,
            "ONB": calc.ONB,
            "LSC": calc.LSC,
            "DBC": calc.DBC,
            "N2": calc.N2,
            "NSG": calc.NSG,
            "ICSV": calc.ICSV,
            "MRCA": calc.MRCA,
            "C1": calc.C1,
            "C2": calc.C2,
            "Purity": calc.purity,
            "neighbourhood_separability": calc.neighbourhood_separability,
        }

        def compute_metric(name):
            try:
                val = measure_map[name]()
                if isinstance(val, (list, tuple, np.ndarray)):
                    if len(val) == 0:
                        return np.nan
                    if name in ["F1"]:
                        return float(np.min(val))
                    elif len(val) == 1:
                        return float(val[0])
                    else:
                        return float(np.mean(val))
                return np.nan if val is None else float(val)
            except Exception:
                return np.nan

        results = [compute_metric(name) for name in features]
        return results, float(np.nanmean(results))
    except Exception:
        return [np.nan] * len(features), np.nan
    finally:
        os.unlink(temp_path)
