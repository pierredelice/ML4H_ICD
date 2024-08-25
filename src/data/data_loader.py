# Read data

import pandas as pd


def read_data(path):
    return pd.read_pickle(path)
