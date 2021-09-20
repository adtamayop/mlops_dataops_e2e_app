import itertools
import os
import sys
import time
from math import exp, log

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

# TODO: Modify project structure for don't do this smell code
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from config.config import DownloadDataParams, Features, Paths
from data.features import Features as feature_engineering


def split_train_test(dataset, perc_train = 0.8, perc_val = 0.1):
    """
    partición de datos
    """
    n = len(dataset)
    train_df = dataset[0:int(n*perc_train)]
    val_df = dataset[int(n*perc_train):int(n*(perc_train+perc_val))]
    test_df = dataset[int(n*(perc_train+perc_val)):]
    return train_df, val_df, test_df

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


def numpy_to_pandas(array, label_array, column_names):
    x_df = pd.DataFrame(
        data=array,
        columns= selected_features
        )
    y_df = pd.DataFrame(
        data=label_array,
        columns=[f"{Features.LABEL_KEY}"]
    )

    return pd.concat([x_df, y_df], axis=1)

def save_dataset(df, type, output_path, company_code):

    folder_path = os.path.join(output_path, company_code, type)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        # print("Output Directory created: ", folder_path)

    df.to_csv(
        os.path.join(
            folder_path, f"{type}.csv"
        ),
        index=False
    )

def save_selected_features(selected_features):

    if not os.path.exists(Paths.SELECTED_FEATURES_DATA_PATH):
        os.makedirs(Paths.SELECTED_FEATURES_DATA_PATH)
        # print("Output Directory created", Paths.SELECTED_FEATURES_DATA_PATH)

    np.save(
    os.path.join(
        Paths.SELECTED_FEATURES_DATA_PATH,
        f"selected_{COMPANY_CODE}_features.npy"
    ),
    selected_features
    )

    print("Selected features stored")


if __name__ == "__main__":

    COMPANY_CODE = DownloadDataParams.COMPANY_CODE

    features = feature_engineering(
        input_data=Paths.RAW_DATA_FILE,
        output_path=Paths.RAW_FEATURES_DATA_PATH,
        company_code = COMPANY_CODE,
        features_file_path = Paths.RAW_FEATURES_DATA_FILE
    )

    df_with_features = features.create_features()
    df_with_features_labelled = features.label_data(df_with_features)
    selected_features = features.feature_selection(df_with_features_labelled)

    save_selected_features(selected_features)

    x_train, x_test, y_train, y_test = train_test_split(
        df_with_features_labelled.loc[:, Features.FIRST_FEATURE:Features.LAST_FEATURE].values,
        df_with_features_labelled['labels'].values, train_size=0.8,
        test_size=0.2, random_state=2, shuffle=True,
        stratify=df_with_features_labelled['labels'].values)

    if 0.7*x_train.shape[0] < 2500:
        train_split = 0.8
    else:
        train_split = 0.7

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train,
        test_size=1-train_split,
        random_state=2, shuffle=True, stratify=y_train)

    print("Splited data")


    # scaler = MinMaxScaler(feature_range=(0, 1))
    # x_train = scaler.fit_transform(x_train)
    # x_val = scaler.transform(x_val)
    # x_test = scaler.transform(x_test)

    # dump(scaler, open(f"{SCALER_FILE}", 'wb'))


    x_train = x_train[:, selected_features]
    x_val = x_val[:, selected_features]
    x_test = x_test[:, selected_features]

    train_dataset = numpy_to_pandas(x_train, y_train, selected_features)
    val_dataset = numpy_to_pandas(x_val, y_val, selected_features)
    test_dataset = numpy_to_pandas(x_test, y_test, selected_features)

    if not os.path.exists(os.path.join(Paths.TRAINING_DATA_PATH, COMPANY_CODE)):
        os.makedirs(os.path.join(Paths.TRAINING_DATA_PATH, COMPANY_CODE))
        # print("Output Directory created", os.path.join(Paths.TRAINING_DATA_PATH, COMPANY_CODE))

    save_dataset(
        df= train_dataset,
        type= "train",
        output_path= Paths.TRAINING_DATA_PATH,
        company_code = COMPANY_CODE
    )

    save_dataset(
            df= val_dataset,
            type= "val",
            output_path= Paths.TRAINING_DATA_PATH,
            company_code = COMPANY_CODE
        )

    save_dataset(
            df= test_dataset,
            type= "test",
            output_path= Paths.TRAINING_DATA_PATH,
            company_code = COMPANY_CODE
        )

    print("Dataset for trainning has been stored")
