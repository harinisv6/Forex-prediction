import pandas as pd

def load_tabular_data(path="data/forex_tabular.csv"):
    data = pd.read_csv(path)
    X = data.drop(columns=["target"])   # target = next day's return/direction
    y = data["target"]
    return X, y

def load_time_series_data(path="data/forex_time_series.csv"):
    df = pd.read_csv(path)
    df["time"] = df["time"].astype(int)
    df["currency_pair"] = df["currency_pair"].astype(str)
    return df
