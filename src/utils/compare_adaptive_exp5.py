import csv

import numpy as np
from scipy.stats import rankdata, ranksums
from tabulate import tabulate

from src.utils.stat import t_test_corrected

ALGORITHM = "KNORAE"
THRESHOLD = 0.8
EXP_SOURCE = "experiment_3"

print(f"\n{'='*80}")
print(
    f"Analysis of results for: {ALGORITHM} (Threshold: {THRESHOLD}, Source: {EXP_SOURCE})"
)
print(f"{'='*80}\n")

datasets_data = {}
with open(
    f"experiments_results/experiment_5/{EXP_SOURCE}/THRESHOLD_{THRESHOLD}/results_{ALGORITHM}.csv"
) as f:
    reader = csv.DictReader(f)
    for row in reader:
        dataset = row["dataset"]
        if dataset not in datasets_data:
            datasets_data[dataset] = {
                "base": [],
                "adaptive_cluster": [],
                "adaptive_knn": [],
            }
        datasets_data[dataset]["base"].append(float(row["base_bacc"]))
        datasets_data[dataset]["adaptive_cluster"].append(
            float(row["adaptive_cluster_bacc"])
        )
        datasets_data[dataset]["adaptive_knn"].append(
            float(row["adaptive_pycol_knn_bacc"])
        )


print(f"\n{'='*80}")
print("Mann-Whitney U test (rank sums) (on average per dataset)")
print(f"{'='*80}\n")

clfs = ["Base", "AdaptiveKNN", "AdaptiveCluster"]
clf_keys = ["base", "adaptive_knn", "adaptive_cluster"]
n_methods = 3
alfa = 0.05

dataset_means = []
for dataset in sorted(datasets_data.keys()):
    dataset_means.append(
        [
            np.mean(datasets_data[dataset]["base"]),
            np.mean(datasets_data[dataset]["adaptive_knn"]),
            np.mean(datasets_data[dataset]["adaptive_cluster"]),
        ]
    )
dataset_means = np.array(dataset_means).T

# Ranking
mean_scores = np.mean(dataset_means, axis=1)
ranks = rankdata(-dataset_means, axis=0)
mean_ranks = np.mean(ranks, axis=1)

print(f"{'Method':<20s} {'Mean BAC':<10s} {'Mean Rank':<10s}")
print("-" * 40)
for i, name in enumerate(clfs):
    print(f"{name:<20s} {mean_scores[i]:<10.4f} {mean_ranks[i]:<10.2f}")

u_statistic = np.zeros((n_methods, n_methods))
p_value = np.zeros((n_methods, n_methods))
for i in range(n_methods):
    for j in range(n_methods):
        if i != j:
            try:
                u_statistic[i, j], p_value[i, j] = ranksums(
                    dataset_means[i], dataset_means[j], alternative="greater"
                )
            except:
                u_statistic[i, j] = 0
                p_value[i, j] = 1.0
        else:
            u_statistic[i, j] = np.nan
            p_value[i, j] = np.nan

names_column = np.array([[name] for name in clfs])

print("\n--- p-value ---")
p_value_table = np.concatenate((names_column, p_value), axis=1)
print(tabulate(p_value_table, headers=clfs, floatfmt=".4f"))

advantage = np.zeros((n_methods, n_methods))
advantage[u_statistic > 0] = 1

significance = np.zeros((n_methods, n_methods))
significance[p_value <= alfa] = 1

stat_better = significance * advantage
print(f"\n--- Statistically Better (α={alfa}, row > column) ---")
stat_better_table = np.concatenate((names_column, stat_better), axis=1)
print(tabulate(stat_better_table, headers=clfs, floatfmt=".0f"))


print(f"\n{'='*80}")
print("Paired corrected t-test per dataset")
print(f"{'='*80}\n")

for dataset in sorted(datasets_data.keys()):
    print(f"\n--- {dataset} ---")

    n_folds = len(datasets_data[dataset]["base"])

    scores = [
        np.mean(datasets_data[dataset]["base"]),
        np.mean(datasets_data[dataset]["adaptive_knn"]),
        np.mean(datasets_data[dataset]["adaptive_cluster"]),
    ]
    print(
        f"Folds: {n_folds}, Results: {clfs[0]}={scores[0]:.4f}, {clfs[1]}={scores[1]:.4f}, {clfs[2]}={scores[2]:.4f}"
    )

    t_statistic = np.zeros((n_methods, n_methods))
    p_value_ttest = np.zeros((n_methods, n_methods))

    for i in range(n_methods):
        for j in range(n_methods):
            if i != j:
                data_i = np.array(datasets_data[dataset][clf_keys[i]])
                data_j = np.array(datasets_data[dataset][clf_keys[j]])
                t_statistic[i, j], p_value_ttest[i, j] = t_test_corrected(
                    data_i, data_j
                )
            else:
                t_statistic[i, j] = np.nan
                p_value_ttest[i, j] = np.nan

    advantage = np.zeros((n_methods, n_methods))
    advantage[t_statistic > 0] = 1

    significance = np.zeros((n_methods, n_methods))
    significance[p_value_ttest <= alfa] = 1

    stat_better = significance * advantage
    stat_better_table = np.concatenate((names_column, stat_better), axis=1)
    print(tabulate(stat_better_table, headers=clfs, floatfmt=".0f"))


print(f"\n{'='*80}")
print("Summary")
print(f"{'='*80}\n")

ranking_order = np.argsort(mean_ranks)
print(f"{'Pos':<5s} {'Method':<20s} {'Ranking':<10s} {'Score':<10s}")
print("-" * 45)
for pos, idx in enumerate(ranking_order, 1):
    print(
        f"{pos:<5d} {clfs[idx]:<20s} {mean_ranks[idx]:<10.2f} {mean_scores[idx]:<10.4f}"
    )

best_idx = ranking_order[0]
print(
    f"\n>>> WINNER: {clfs[best_idx]} (ranking: {mean_ranks[best_idx]:.2f}, score: {mean_scores[best_idx]:.4f})"
)
