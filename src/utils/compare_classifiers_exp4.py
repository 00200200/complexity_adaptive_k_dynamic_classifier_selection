import csv
import sys

import numpy as np
from scipy.stats import rankdata, wilcoxon
from tabulate import tabulate

if len(sys.argv) > 1:
    experiment = sys.argv[1]
else:
    experiment = "experiment_3"

print(f"\n{'='*80}")
print(f"Analysis of results for: {experiment}")
print(f"{'='*80}\n")

detailed = {}
with open(f"experiments_results/experiment_4/{experiment}/detailed_results.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        alg = row["algorithm"]
        clf_name = row["classifier"]
        sampling = row["sampling"]
        bacc_test = float(row["bacc_test"])

        if alg not in detailed:
            detailed[alg] = {}

        key = f"{clf_name}_{sampling}"
        if key not in detailed[alg]:
            detailed[alg][key] = {"scores": [], "clf": clf_name, "sampling": sampling}

        detailed[alg][key]["scores"].append(bacc_test)

for alg in sorted(detailed.keys()):
    print(f"\n{'='*80}")
    print(f"Algorithm: {alg}")
    print(f"{'='*80}")

    configs = detailed[alg]
    if not configs:
        print("  No configurations!")
        continue

    sorted_configs = sorted(
        configs.items(), key=lambda x: np.mean(x[1]["scores"]), reverse=True
    )
    all_scores_matrix = np.array([data["scores"] for _, data in sorted_configs])

    n_folds = all_scores_matrix.shape[1]
    config_means = np.mean(all_scores_matrix, axis=1)

    all_ranks = rankdata(-all_scores_matrix, axis=0)
    all_mean_ranks = np.mean(all_ranks, axis=1)

    random_baseline_idx = None
    random_baseline_key = None
    for idx, (key, data) in enumerate(sorted_configs):
        if "Random_none" in key:
            random_baseline_idx = idx
            random_baseline_key = key
            break

    rank_order = np.argsort(-config_means)
    top_5_indices = rank_order[:5]

    if random_baseline_idx is not None and random_baseline_idx not in top_5_indices:
        top_indices = np.append(top_5_indices, random_baseline_idx)
    else:
        top_indices = top_5_indices

    print(
        f"\nTop 5 classifiers + Random baseline (by average score on all {len(sorted_configs)} configurations):"
    )
    for i, idx in enumerate(top_indices, 1):
        key, data = sorted_configs[idx]
        scores = data["scores"]
        marker = " [BASELINE]" if "Random_none" in key else ""
        print(
            f"  {i}. {key:40s}: {np.mean(scores):.4f} ± {np.std(scores):.4f} (rank: {all_mean_ranks[idx]:.2f}){marker}"
        )

    top_configs = [sorted_configs[i] for i in top_indices]
    n_clfs = len(top_configs)
    clf_names = [
        f"{data['clf'][:15]}_{data['sampling'][:4]}" for _, data in top_configs
    ]
    scores_matrix = np.array([data["scores"] for _, data in top_configs])

    mean_scores = np.mean(scores_matrix, axis=1)
    std_scores = np.std(scores_matrix, axis=1)

    mean_ranks = np.array([all_mean_ranks[i] for i in top_indices])

    print(f"\n{'Classifier':<25s} {'Mean':<10s} {'Std':<10s} {'Mean Rank':<10s}")
    print("-" * 55)
    for i, name in enumerate(clf_names):
        print(
            f"{name:<25s} {mean_scores[i]:<10.4f} {std_scores[i]:<10.4f} {mean_ranks[i]:<10.2f}"
        )

    alfa = 0.05

    n_folds = scores_matrix.shape[1]

    print(f"\n[INFO] Detected {n_folds} folds (Leave-One-Group-Out).")
    print("[INFO] Using Wilcoxon signed-rank test across folds (datasets).")

    p_value_wilcoxon = np.zeros((n_clfs, n_clfs))

    for i in range(n_clfs):
        for j in range(n_clfs):
            if i != j:
                diffs = scores_matrix[i] - scores_matrix[j]
                if np.allclose(diffs, 0):
                    p_val = 1.0
                else:
                    try:
                        p_val = wilcoxon(
                            scores_matrix[i],
                            scores_matrix[j],
                            zero_method="pratt",
                            alternative="two-sided",
                        ).pvalue
                    except ValueError:
                        p_val = 1.0
                p_value_wilcoxon[i, j] = p_val
            else:
                p_value_wilcoxon[i, j] = np.nan

    advantage_wilcoxon = np.zeros((n_clfs, n_clfs))
    for i in range(n_clfs):
        for j in range(n_clfs):
            if i != j:
                if np.mean(scores_matrix[i] - scores_matrix[j]) > 0:
                    advantage_wilcoxon[i, j] = 1

    significance_wilcoxon = np.zeros((n_clfs, n_clfs))
    significance_wilcoxon[p_value_wilcoxon <= alfa] = 1

    stat_better_wilcoxon = significance_wilcoxon * advantage_wilcoxon

    names_column = np.array([[name] for name in clf_names])

    print(f"\n{'='*80}")
    print("Wilcoxon signed-rank test (paired)")
    print(f"{'='*80}")

    print("\n--- Statistically Better ---")
    better_table = np.concatenate((names_column, stat_better_wilcoxon), axis=1)
    print(tabulate(better_table, headers=clf_names, floatfmt=".0f"))

    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")

    wins_wilcoxon = np.sum(stat_better_wilcoxon, axis=1)
    losses_wilcoxon = np.sum(stat_better_wilcoxon, axis=0)

    print("\n--- Wilcoxon ---")
    print(f"{'Classifier':<25s} {'Wins':<10s} {'Losses':<10s}")
    print("-" * 45)
    for i, name in enumerate(clf_names):
        print(
            f"{name:<25s} {int(wins_wilcoxon[i]):<10d} {int(losses_wilcoxon[i]):<10d}"
        )

    print(f"\n{'='*80}")
    print("Ranking (by average rank, lower rank = better)")
    print(f"{'='*80}")

    ranking_order = np.argsort(mean_ranks)

    print(f"\n{'Pos':<5s} {'Classifier':<25s} {'Mean Rank':<12s} {'Score':<10s}")
    print("-" * 55)
    for pos, idx in enumerate(ranking_order, 1):
        print(
            f"{pos:<5d} {clf_names[idx]:<25s} {mean_ranks[idx]:<12.2f} {mean_scores[idx]:<10.4f}"
        )

    best_idx = ranking_order[0]
    print(
        f"\n>>> WINNER: {clf_names[best_idx]} (ranking: {mean_ranks[best_idx]:.2f}, score: {mean_scores[best_idx]:.4f})"
    )
