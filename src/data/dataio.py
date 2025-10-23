import pandas as pd


def load_data(dataset_name):
    df = pd.read_csv(f"data/{dataset_name}.csv")
    return df
