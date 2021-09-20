import os

import pandas as pd


class DownloadDataParams:
    API_KEY = "I8H8UHPZGYABYJL1"
    COMPANY_CODE = "WMT" # "NDAQ" #
    DATA_TYPE = "csv"
    API_FUNCTION = "TIME_SERIES_DAILY_ADJUSTED"
    TRAINING_OUT_SIZE = "full"
    BASE_URL = f"https://www.alphavantage.co/query?function={API_FUNCTION}&outputsize={TRAINING_OUT_SIZE}&apikey={API_KEY}&datatype={DATA_TYPE}&symbol={COMPANY_CODE}"

class Paths:
    ROOT_DIR = os.path.abspath(os.curdir)
    ROOT_DATA_PATH = os.path.join(ROOT_DIR,"data")
    RAW_DATA_PATH = os.path.join(ROOT_DATA_PATH, "raw")
    RAW_FEATURES_DATA_PATH = os.path.join(ROOT_DATA_PATH, "raw_features")
    RAW_LABELLED_DATA_PATH = os.path.join(ROOT_DATA_PATH, "raw_labelled")
    SELECTED_FEATURES_DATA_PATH = os.path.join(ROOT_DATA_PATH, "selected_features")
    RAW_DATA_FILE = os.path.join(RAW_DATA_PATH,f"raw_{DownloadDataParams.COMPANY_CODE}.csv")
    RAW_FEATURES_DATA_FILE = os.path.join(RAW_FEATURES_DATA_PATH,f"raw_features_{DownloadDataParams.COMPANY_CODE}.csv")
    RAW_LABELLED_DATA_FILE = os.path.join(RAW_LABELLED_DATA_PATH,f"raw_labelled_{DownloadDataParams.COMPANY_CODE}.csv")


    TRAINING_DATA_PATH = os.path.join(ROOT_DATA_PATH, "train_data")
    TEST_DATA_PATH = os.path.join(ROOT_DATA_PATH, "test_data")
    LAST_DATA_PATH = os.path.join(ROOT_DATA_PATH, "last_data")


    RAW_LAST_DATA_FILE = os.path.join(LAST_DATA_PATH,f"raw_last_{DownloadDataParams.COMPANY_CODE}.csv")
    LAST_DATA_FEATURES_FILE = os.path.join(LAST_DATA_PATH,f"raw_last_features_{DownloadDataParams.COMPANY_CODE}.csv")

    LAST_LABELLED_DATA_FILE = os.path.join(LAST_DATA_PATH,f"raw_last_labelled_{DownloadDataParams.COMPANY_CODE}.csv")
    # DATA_PATH = "data/"
    # DATA_RAW_PATH = f"{DATA_PATH}raw/"
    # DATA_RAW_BY_MONTH = f"{DATA_RAW_PATH}separated/"
    # DATA_RAW_JOIN_PATH = f"{DATA_RAW_PATH}joined/"
    # CONFIG_PARAMS_PATH = "./src/config/params.json"
    # PIPELINE_PATH = "./tfx_pipeline/"
    # JOIN_NAME_DATASET = "raw_joined_dataset.csv"
    # PATH_LABELLED_DATA = f"{DATA_PATH}processed/labeled/"
    # LABELLED_DATA_NAME = "labelled_dataset.csv"
    # SPLITTED_DATA_PATH = f"{DATA_PATH}processed/split/"
    # DATA_TEST_PATH = f"{DATA_PATH}test_dataset/"
    # DATA_TEST_PATH_FILE = f"{DATA_TEST_PATH}train/train.csv"

class Features:
    STRATEGY_TYPE = "original"
    N_FEATURES = 225
    FIRST_FEATURE = "open"
    LAST_FEATURE = "eom_26"

    DATE_FEATURE_NAME = "timestamp"
    RAW_INPUT_FEATURES = ["timestamp","open","high","low","close","adjusted_close","volume","dividend_amount","split_coefficient"]
    RAW_FEATURES = ["timestamp", "open", "high", "low", "close", "volume"]
    # RAW_INPUT_LABEL = "close"
    # LABEL_KEY = "label"
    LABEL_KEY = "labels"
    _df_train = pd.read_csv(
        os.path.join(
           Paths.TRAINING_DATA_PATH,
           DownloadDataParams.COMPANY_CODE,
           "train",
           "train.csv"
        )
    )
    FEATURE_KEYS = list(_df_train.columns.values)
    FEATURE_KEYS.remove(LABEL_KEY)

# class LabellingParams:
#     # UP = 0
#     # DOWN = 1
#     # STATIONARY = 2
#     # ROLLING_AVG_WINDOW_SIZE = 20
#     # STATIONARY_THRESHOLD = 0.0001
#     # SHIFT = -(ROLLING_AVG_WINDOW_SIZE-1)
#     # N_TSTEPS = 20

# class ModelConstants:
#     HIDDEN_LAYER_UNITS = 50
#     OUTPUT_LAYER_UNITS = 3
#     NUM_LAYERS = 10
#     LEARNING_RATE = 0.01
#     TRAIN_BATCH_SIZE = 30
#     EVAL_BATCH_SIZE = 20
#     EPOCHS = 10


# # TIME SERIES PARAMETERS
# LABEL_NAME = "close"
# PREVIOUS_STEPS = 15


# if __name__ == "__main__":

#     print(Features.FEATURE_KEYS)
