import os
import sys

import numpy as np
import pandas as pd

# TODO: Modify project structure for don't do this smell code
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from config.config import Features, LabellingParams, Paths


def label_row(row):
    if row[f"{Features.RAW_INPUT_LABEL}_avg_change_pct"] > LabellingParams.STATIONARY_THRESHOLD:
        return LabellingParams.UP
    elif row[f'{Features.RAW_INPUT_LABEL}_avg_change_pct'] < -LabellingParams.STATIONARY_THRESHOLD:
        return LabellingParams.DOWN
    else:
        return LabellingParams.STATIONARY

def labelling_time_series(data, features, label):
    data = data[features]
    data[f'{label}_avg'] = data[f'{label}']\
        .rolling(window=LabellingParams.ROLLING_AVG_WINDOW_SIZE)\
        .mean()\
        .shift(LabellingParams.SHIFT)
    data[f'{label}_avg_change_pct'] = (data[f'{label}_avg'] - data[f'{label}']) / data[f'{label}']
    data[f'{Features.LABEL_KEY}'] = data.apply(label_row, axis=1)
    return data

def supervised_data(dataset_labelled):
    """

    """
    data = []
    labels = []
    for i in range(len(dataset_labelled) - LabellingParams.N_TSTEPS + 1 + LabellingParams.SHIFT):
        label = dataset_labelled[f'{Features.LABEL_KEY}'].iloc[i+LabellingParams.N_TSTEPS - 1]
        data.append(dataset_labelled[Features.RAW_INPUT_FEATURES].iloc[i:i+LabellingParams.N_TSTEPS].values)
        labels.append(label)
    return np.array(data), labels

def save_data_as_csv(dataset, labels):

    dataset_flattened = pd.DataFrame(
        dataset.reshape(dataset.shape[0],
        dataset.shape[1]*dataset.shape[2]),
        columns= [f"col_t_{i}" for i in range(dataset.shape[1]*dataset.shape[2])])

    labels = pd.DataFrame(columns=[f"{Features.LABEL_KEY}"], data=labels)
    dataset = pd.concat([dataset_flattened, labels], axis=1)
    dataset.to_csv(
        index=False,
        path_or_buf=f"{Paths.PATH_LABELLED_DATA}{Paths.LABELLED_DATA_NAME}")

if __name__ == "__main__":
    data = pd.read_csv(f"{Paths.DATA_RAW_JOIN_PATH}/{Paths.JOIN_NAME_DATASET}")
    features = Features.RAW_INPUT_FEATURES
    label = Features.RAW_INPUT_LABEL

    data_labelled = labelling_time_series(data, features, label)
    dataset, label = supervised_data(data_labelled)
    save_data_as_csv(dataset, label)
