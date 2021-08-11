
import json
import os
import sys

# TODO: code smell of project estructure
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

from src.config.config import DownloadDataParams


def test_months_to_download():
    month = DownloadDataParams.MONTHS_TO_DOWNLOAD
    assert(month>0 and month<13)

def test_years_to_download():
    year = month = DownloadDataParams.YEARS_TO_DOWNLOAD
    assert(year>0 and year<3)

if __name__ == "__main__":
    test_months_to_download()
    test_years_to_download()
