import numpy as np

from src.data.dataio import load_data
from src.utils.utils import get_dataset_names


def analyze_dataset(name):
    df = load_data(name)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    n_classes = len(np.unique(y))
    y_binary = (y == 1.0).astype(int)
    return (
        len(X),
        X.shape[1],
        int((y_binary == 0).sum()),
        int((y_binary == 1).sum()),
        n_classes,
    )


if __name__ == "__main__":
    datasets = get_dataset_names()
    print(
        f"{'Dataset':<12} {'Samples':>8} {'Features':>8} {'Class 0':>10} {'Class 1':>10} {'N classes':>8}"
    )
    print("-" * 65)
    for dataset_name in datasets:
        n_samples, n_features, n_class_0, n_class_1, n_classes = analyze_dataset(
            dataset_name
        )
        print(
            f"{dataset_name:<12} {n_samples:>8} {n_features:>8} {n_class_0:>10} {n_class_1:>10} {n_classes:>8}"
        )
