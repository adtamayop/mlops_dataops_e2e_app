import pandas as pd

class DownloadDataParams:
    # CONSTANTS OF DATA DOWNLOAD
    MONTHS_TO_DOWNLOAD = 12
    YEARS_TO_DOWNLOAD= 2
    FUNCTION= "TIME_SERIES_INTRADAY_EXTENDED"
    SYMBOL= "NDAQ"
    INTERVAL= "30min"
    DONWLOAD_NEW_DATA= True
    JOIN_SEPARATED_DATA = True
    BASE_URL = "https://www.alphavantage.co/query?"
    TIME_SLEEP_DOWNLOAD_DATA = 13
    ALPHAVANTAGE_API_KEY = "I8H8UHPZGYABYJL1"

class Paths:
    DATA_PATH = "data/"
    DATA_RAW_PATH = f"{DATA_PATH}raw/"
    DATA_RAW_BY_MONTH = f"{DATA_RAW_PATH}separated/"
    DATA_RAW_JOIN_PATH = f"{DATA_RAW_PATH}joined/"
    CONFIG_PARAMS_PATH = "./src/config/params.json"
    PIPELINE_PATH = "./tfx_pipeline/"
    JOIN_NAME_DATASET = "raw_joined_dataset.csv"
    PATH_LABELLED_DATA = f"{DATA_PATH}processed/labeled/"
    LABELLED_DATA_NAME = "labelled_dataset.csv"
    SPLITTED_DATA_PATH = f"{DATA_PATH}processed/split/"

class Features:
    RAW_INPUT_FEATURES = ["open", "high", "low", "close", "volume"]
    RAW_INPUT_LABEL = "close"
    LABEL_KEY = "label"
    df_train = pd.read_csv("data/test/data.csv")
    FEATURE_KEYS = list(df_train.columns.values)
    FEATURE_KEYS.remove(LABEL_KEY)

class LabellingParams:
    UP = 0
    DOWN = 1
    STATIONARY = 2
    ROLLING_AVG_WINDOW_SIZE = 20
    STATIONARY_THRESHOLD = 0.0001
    SHIFT = -(ROLLING_AVG_WINDOW_SIZE-1)
    N_TSTEPS = 20





# TIME SERIES PARAMETERS
LABEL_NAME = "close"
PREVIOUS_STEPS = 15


if __name__ == "__main__":

    print(Features.FEATURE_KEYS)