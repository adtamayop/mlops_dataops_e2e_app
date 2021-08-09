import glob
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# from dotenv import load_dotenv

# TODO: Modify project structure for don't do this smell code
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from config.config import DownloadDataParams as params
from config.config import Paths


def get_months_and_years_to_query(months, years):
    """

    """
    months = [month for month in range(1, (months + 1))]
    years = [year for year in range(1, (years + 1))]
    month_year = [months, years]
    month_year = list(itertools.product(*month_year))
    return sorted(month_year, key=lambda tup: tup[1])

def delete_existing_data(path_data, window_date):
    """

    """
    files = glob.glob(f"{path_data}{window_date}")
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(e)

def download_raw_data(parameters, api_key, dates_sorted):
    """

    """
    for month, year in dates_sorted:
        window_date = f"year{year}month{month}"

        api_parameters = {
            "function": parameters.FUNCTION,
            "symbol": parameters.SYMBOL,
            "interval": parameters.INTERVAL,
            "slice": window_date,
            "apikey": api_key,
        }

        print(f"Getting data for {window_date}")
        response = requests.get(params.BASE_URL, params=api_parameters)
        try:
            response_content = response.content
            delete_existing_data(Paths.DATA_RAW_BY_MONTH, window_date)
            with open(f"{Paths.DATA_RAW_BY_MONTH}{window_date}.csv", "wb") as csv_file:
                csv_file.write(response_content)
            time.sleep(params.TIME_SLEEP_DOWNLOAD_DATA)
        except Exception as e:
            print(f"Error al obtener el contenido {e}")

def request_symbols(token):
    """

    """
    q_string = "https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={}"
    print("Retrieving stock symbols from Alpha Vantage...")
    r = requests.get(q_string.format(token)).content
    print("Data has been successfully downloaded...")
    r = r.decode("utf-8")
    colnames = list(range(0, 6))
    df = pd.DataFrame(columns=colnames)
    print("Sorting the retrieved data into a dataframe...")
    for i in tqdm(range(1, len(r.split("\r\n")) - 1)):
        row = pd.DataFrame(r.split("\r\n")[i].split(",")).T
        df = pd.concat([df, row], ignore_index=True)
    df.columns = r.split("\r\n")[0].split(",")
    return df


if __name__ == "__main__":
    # TODO
    # load_dotenv()
    # api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    api_key = params.ALPHAVANTAGE_API_KEY
    dates_sorted = get_months_and_years_to_query(
        params.MONTHS_TO_DOWNLOAD,
        params.YEARS_TO_DOWNLOAD)

    if params.DONWLOAD_NEW_DATA:
        download_raw_data(params, api_key, dates_sorted)
