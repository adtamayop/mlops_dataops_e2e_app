import os
import re
import sys
import urllib.request as request

import pandas as pd
import requests

# TODO: Modify project structure for don't do this smell code
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from config.config import DownloadDataParams, Paths

# from src.config.config.Paths import RAW_DATA_PATH, RAW_DATA_FILE
# from src.config.config.DownloadDataParams import COMPANY_CODE, BASE_URL

class Download_raw_data:

    def __init__(self, company_code, output_path, url, file_path):
        self.company_code = company_code
        self.output_path = output_path
        self.url = url
        self.file_path = file_path

        print(f"[INFO] Start download data process for {self.company_code} company")
        self.download_stock_data()

    def download_stock_data(self):
        """
        Download data for the AlphaVentage API and
        save .csv in data path
        """
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            print("Output Directory created", self.output_path)

        print("[INFO] Path to company data:", self.output_path)
        print("[INFO] Starting download " + re.sub(r'apikey=[A-Za-z0-9]+&', 'apikey=my_api_key&', self.url))
        request.urlretrieve(self.url, self.file_path)
        print("[INFO] Finish download data")

# TODO: meter dentro de la clase
def download_last_data(data_path, company_code):

    df_complete = pd.DataFrame(
        {'time': pd.Series([], dtype='object'),
        'open': pd.Series([], dtype='float64'),
        'high': pd.Series([], dtype='float64'),
        'low': pd.Series([], dtype='float64'),
        'close': pd.Series([], dtype='float64'),
        'volume': pd.Series([], dtype='int64')})

    api_key = "I8H8UHPZGYABYJL1"
    base_url = 'https://www.alphavantage.co/query?'


    params = {'function': 'TIME_SERIES_DAILY',
            'symbol': f'{company_code}',
            'datatype': 'csv',
            'outputsize':'compact',
            'apikey': api_key}
    print(f"Getting last 100 data points")
    response = requests.get(base_url, params=params)

    try:
        response_content = response.content
    except Exception as e:
        print(f"Error al obtener el contenido {e}")

    try:
        os.remove(f'{data_path}/raw_last_{company_code}.csv')
        #print("Borrado con exito")
    except OSError:
        # TODO! separar las excepciones
        # print("No se pudo borrar")
        pass

    try:
        with open(f'{data_path}/raw_last_{company_code}.csv', 'wb') as csv_file:
            csv_file.write(response_content)
    except:
        print("Error al escribir archivo")


if __name__ == "__main__":

    Download_raw_data(
        company_code = DownloadDataParams.COMPANY_CODE,
        output_path = Paths.RAW_DATA_PATH,
        url=DownloadDataParams.BASE_URL,
        file_path = Paths.RAW_DATA_FILE
    )
