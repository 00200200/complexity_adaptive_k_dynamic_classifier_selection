import os
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.utils.utils import set_determinism

warnings.filterwarnings("ignore")

ALGORITHM_FEATURES_EXP3 = {
    "DESMI": [
        "F1",
        "F2",
        "input_noise",
        "N2",
        "MRCA",
        "C2",
        "neighbourhood_separability",
    ],
    "OLA": ["D3_value", "LSC", "Purity"],
    "KNORAE": ["Purity", "D3_value", "LSC"],
    "KNORAU": ["F3", "F4", "N4"],
    "Rank": [
        "R_value",
        "degOver",
        "N3",
        "SI",
        "kDN",
        "D3_value",
        "CM",
        "borderline",
        "N1",
        "Clst",
        "ONB",
        "LSC",
        "DBC",
        "NSG",
        "C1",
        "Purity",
    ],
}


def get_classifiers():
    classifiers = []

    classifiers.append(("Random", DummyClassifier(strategy="uniform", random_state=42)))

    for n_estimators in [50, 100, 150]:
        classifiers.append(
            (
                f"RF_{n_estimators}",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=3,
                    min_samples_split=10,
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        )
        classifiers.append(
            (
                f"XGB_{n_estimators}",
                XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=3,
                    min_child_weight=10,
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        )

    classifiers.append(
        ("SVC_Linear", SVC(kernel="linear", probability=True, random_state=42))
    )
    classifiers.append(
        ("SVC_RBF", SVC(kernel="rbf", probability=True, random_state=42))
    )
    classifiers.append(("GNB", GaussianNB()))

    classifiers.append(
        (
            "MLP",
            MLPClassifier(
                hidden_layer_sizes=(50, 50), random_state=42, early_stopping=True
            ),
        )
    )

    return classifiers


SAMPLING = {
    "none": None,
    "smote": SMOTE(random_state=42),
    "undersample": RandomUnderSampler(random_state=42),
}

SAMPLING_CLASS_WEIGHT = {
    "none": "balanced",
    "smote": None,
    "undersample": None,
}


def test_model(clf_name, clf, sampler, class_weight, X, y, groups, cv):
    from sklearn.base import clone

    train_scores, test_scores, test_groups = [], [], []

    for train_idx, test_idx in cv.split(X, y, groups):
        X_train, X_test, y_train, y_test = (
            X[train_idx],
            X[test_idx],
            y[train_idx],
            y[test_idx],
        )
        test_group = groups[test_idx][0] if len(groups[test_idx]) > 0 else "unknown"

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        if sampler:
            try:
                X_train, y_train = sampler.fit_resample(X_train, y_train)
            except Exception:
                pass

        model = clone(clf)
        if class_weight == "balanced":
            if clf_name in ["GaussianNB", "MLP", "Random"]:
                model.fit(X_train, y_train)
            else:
                sample_weights = compute_sample_weight("balanced", y_train)
                try:
                    model.fit(X_train, y_train, sample_weight=sample_weights)
                except TypeError:
                    model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        train_scores.append(balanced_accuracy_score(y_train, model.predict(X_train)))
        test_scores.append(balanced_accuracy_score(y_test, model.predict(X_test)))
        test_groups.append(test_group)

    return train_scores, test_scores, test_groups


def prepare_data(df, algorithm, algorithm_features):
    target_col = f"{algorithm}_optimal_k"

    available_features = [
        f for f in algorithm_features.get(algorithm, []) if f in df.columns
    ]
    if not available_features:
        return None, None, None, None, None

    X = df[available_features].values
    y = df[target_col].values
    groups = df["dataset"].values if "dataset" in df.columns else np.arange(len(df))

    unique_classes = np.unique(y)
    class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
    y_encoded = np.array([class_mapping[val] for val in y])

    return X, y_encoded, groups, available_features, unique_classes


def run_for_algorithm(experiment_name, algorithm, algorithm_features):
    csv_path = f"data/generated_datasets/{experiment_name}/{algorithm}.csv"

    if not os.path.exists(csv_path):
        print(f"Skipped {algorithm} - file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    target_col = f"{algorithm}_optimal_k"

    if target_col not in df.columns:
        print(f"Skipped {algorithm} - missing column: {target_col}")
        return None

    X, y_encoded, groups, available_features, unique_classes = prepare_data(
        df, algorithm, algorithm_features
    )

    if X is None:
        print(f"Skipped {algorithm} - no features available")
        return None

    unique_groups = np.unique(groups)
    print(f"\n{'='*80}")
    print(f"Algorithm: {algorithm}")
    print(f"Features ({len(available_features)}): {available_features}")
    print(f"Samples: {len(X)}, Classes: {len(unique_classes)} {unique_classes}")
    print(f"Groups: {len(unique_groups)}")

    cv = LeaveOneGroupOut()

    detailed_results, summary_results = [], []
    config_id = 0

    for clf_name, clf in get_classifiers():
        for samp_name, sampler in SAMPLING.items():
            if clf_name == "Random" and samp_name != "none":
                continue

            config_id += 1
            class_weight = SAMPLING_CLASS_WEIGHT[samp_name]
            train_scores, test_scores, test_groups = test_model(
                clf_name, clf, sampler, class_weight, X, y_encoded, groups, cv
            )

            for fold_id, (bacc_train, bacc_test, test_group) in enumerate(
                zip(train_scores, test_scores, test_groups)
            ):
                detailed_results.append(
                    {
                        "algorithm": algorithm,
                        "config_id": config_id,
                        "classifier": clf_name,
                        "sampling": samp_name,
                        "class_weight": str(class_weight),
                        "fold": fold_id,
                        "test_dataset": test_group,
                        "bacc_train": bacc_train,
                        "bacc_test": bacc_test,
                    }
                )

            mean_train, std_train = np.mean(train_scores), np.std(train_scores)
            mean_test, std_test = np.mean(test_scores), np.std(test_scores)

            summary_results.append(
                {
                    "algorithm": algorithm,
                    "config_id": config_id,
                    "classifier": clf_name,
                    "sampling": samp_name,
                    "class_weight": str(class_weight),
                    "bacc_train_mean": mean_train,
                    "bacc_train_std": std_train,
                    "bacc_test_mean": mean_test,
                    "bacc_test_std": std_test,
                }
            )

            print(
                f"  {config_id:3d} {clf_name:20s} {samp_name:12s} test={mean_test:.4f}±{std_test:.4f} train={mean_train:.4f}"
            )

    return {
        "algorithm": algorithm,
        "features": available_features,
        "n_samples": len(X),
        "n_classes": len(unique_classes),
        "classes": unique_classes.tolist(),
        "detailed": detailed_results,
        "summary": summary_results,
    }


def main():
    experiment_name = "experiment_3"
    ALGORITHM_FEATURES = ALGORITHM_FEATURES_EXP3
    algorithms = list(ALGORITHM_FEATURES.keys())

    data_dir = f"data/generated_datasets/{experiment_name}"
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist")
        return None

    print(f"Experiment: {experiment_name}")
    print(f"Algorithms to test: {algorithms}")

    all_detailed = []
    all_summary = []

    for alg in algorithms:
        result = run_for_algorithm(experiment_name, alg, ALGORITHM_FEATURES)
        if result:
            all_detailed.extend(result["detailed"])
            all_summary.extend(result["summary"])

    if not all_summary:
        print("No results to save")
        return None

    out_dir = f"experiments_results/experiment_4/{experiment_name}"
    os.makedirs(out_dir, exist_ok=True)

    df_detailed = pd.DataFrame(all_detailed)
    df_summary = pd.DataFrame(all_summary)

    df_detailed.to_csv(f"{out_dir}/detailed_results.csv", index=False)
    df_summary.to_csv(f"{out_dir}/summary.csv", index=False)

    print(f"\n{'='*80}")
    print("SUMMARY - Best configurations for each algorithm:")
    print(f"{'='*80}")

    for alg in df_summary["algorithm"].unique():
        alg_results = df_summary[df_summary["algorithm"] == alg].sort_values(
            "bacc_test_mean", ascending=False
        )
        best = alg_results.iloc[0]
        print(f"\n{alg}:")
        print(f"  Classifier: {best['classifier']}")
        print(f"  Sampling: {best['sampling']}")
        print(
            f"  Train bacc: {best['bacc_train_mean']:.4f}±{best['bacc_train_std']:.4f}"
        )
        print(f"  Test bacc:  {best['bacc_test_mean']:.4f}±{best['bacc_test_std']:.4f}")

    print(f"\nResults saved to: {out_dir}/")
    print(f"  - detailed_results.csv: {len(df_detailed)} rows (all folds)")
    print(f"  - summary.csv: {len(df_summary)} rows (means)")

    return df_summary


if __name__ == "__main__":
    set_determinism(42)
    main()
