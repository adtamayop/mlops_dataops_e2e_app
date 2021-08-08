import os
import sys
import time
import requests
import itertools
import numpy as np
import pandas as pd 
from tqdm import tqdm

from math import log, exp
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# TODO: Modify project structure for don't do this smell code
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from config.config import Paths

def split_train_test(dataset, perc_train = 0.8, perc_val = 0.1):
    """
    partición de datos
    """
    n = len(dataset)
    train_df = dataset[0:int(n*perc_train)]
    val_df = dataset[int(n*perc_train):int(n*(perc_train+perc_val))]
    test_df = dataset[int(n*(perc_train+perc_val)):]
    return train_df, val_df, test_df

### Normalización
def normalize_data(data, type="train", scaler=None):
    """
    """
    # TODO: save scaler to disk to later predictions in real time
    # Convert to clasess to separate methods (maybe mix norm and stand)
    if type=="train":
        scaler = MinMaxScaler()
        return scaler.fit_transform(data), scaler
    elif type=="val" or type=="test":
        return scaler.transform(data)
    elif type=="inverse":
        return scaler.inverse_transform(data)

"""### Estandarización"""
def standarize_data(data, type="train", scaler=None):
    """

    """
    # TODO: save scaler to disk to later predictions in real time
    if type=="train":
        scaler = StandardScaler()
        return scaler.fit_transform(data), scaler
    elif type=="val" or type=="test":
        return scaler.transform(data)
    elif type=="inverse":
        return scaler.inverse_transform(data)


if __name__ == "__main__":
    data = pd.read_csv(f"{Paths.PATH_LABELLED_DATA}/{Paths.LABELLED_DATA_NAME}")
    train_df, val_df, test_df = split_train_test(data)
    train_df.to_csv(f"{Paths.SPLITTED_DATA_PATH}/train/train.csv", index=False)
    val_df.to_csv(f"{Paths.SPLITTED_DATA_PATH}/val/val.csv", index=False)
    test_df.to_csv(f"{Paths.SPLITTED_DATA_PATH}/test/test.csv", index=False)
    