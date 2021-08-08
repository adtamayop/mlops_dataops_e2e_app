import os
import sys
import pandas as pd

# TODO: Modify project structure for don't do this smell code
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from config.config import Paths
from config.config import DownloadDataParams as params
from data.get_external_data import get_months_and_years_to_query


def join_data(dates_sorted):
    """
    """
    #!TODO: esta estructura debería ser dinámica o cargada de alguna parte
    complete_dataset = pd.DataFrame(
        {
            "time": pd.Series([], dtype="object"),
            "open": pd.Series([], dtype="float64"),
            "high": pd.Series([], dtype="float64"),
            "low": pd.Series([], dtype="float64"),
            "close": pd.Series([], dtype="float64"),
            "volume": pd.Series([], dtype="int64"),
        }
    )
    for month, year in dates_sorted:
        window_date = f"year{year}month{month}"
        try:
            print(f"Reading data of {window_date}")
            mes = pd.read_csv(f"{Paths.DATA_RAW_BY_MONTH}{window_date}.csv")
            complete_dataset = pd.concat(
                [complete_dataset, mes], axis=0, ignore_index=True
            )
            complete_dataset.reset_index(drop=True)
        except OSError:
            print(f"Error al unir el dataset {window_date}")

    df_complete_reverse = complete_dataset.iloc[::-1]
    df_complete_reverse = df_complete_reverse.reset_index(drop=True, inplace=False)
    return df_complete_reverse

def write_dataset(dataset):
    try:
        os.remove(f"{Paths.DATA_RAW_JOIN_PATH}{Paths.JOIN_NAME_DATASET}")
        dataset.to_csv(f"{Paths.DATA_RAW_JOIN_PATH}{Paths.JOIN_NAME_DATASET}", index=False)
    except OSError:
        dataset.to_csv(f"{Paths.DATA_RAW_JOIN_PATH}{Paths.JOIN_NAME_DATASET}", index=False)

if __name__ == "__main__":
    if params.JOIN_SEPARATED_DATA:
        dates_sorted = get_months_and_years_to_query(
            params.MONTHS_TO_DOWNLOAD, 
            params.YEARS_TO_DOWNLOAD)

        data = join_data(dates_sorted)
        write_dataset(data)