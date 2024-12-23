import pandas as pd
import numpy as np

def load_data(filepaths):
    data_frames = [pd.read_excel(fp) for fp in filepaths]
    for df in data_frames:
        df.set_index('Date', inplace=True)
    return pd.concat(data_frames, axis=1).dropna()

def preprocess_data(data):
    data = data.pct_change().dropna()
    return data
