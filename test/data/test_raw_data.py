import filecmp
import itertools
import os
import sys
from os import listdir
from os.path import isfile, join

import pandas as pd
from numpy.lib.arraysetops import unique

# TODO: code smell of project estructure
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

from src.config.config import Features, Paths


def test_size_file():
    all_files = listdir(Paths.RAW_DATA_PATH)
    for file in all_files:
        size = os.path.getsize(os.path.join(Paths.RAW_DATA_PATH, file))
        assert(size>0)

def test_unique_date():
    df = pd.read_csv(f"{Paths.RAW_DATA_FILE}")
    unique_len = len(df[f"{Features.DATE_FEATURE_NAME}"].unique())
    set_len = len(set(df[f"{Features.DATE_FEATURE_NAME}"].values))
    assert(unique_len==set_len)

def test_rows():
    df = pd.read_csv(f"{Paths.RAW_DATA_FILE}")
    assert(df.shape[0]>2000)

def test_unique_content():
    path_raw = Paths.RAW_DATA_PATH
    all_files = listdir(path_raw)
    couple_combination = list(itertools.combinations(all_files, 2))
    for a,b in couple_combination:
        df_a = pd.read_csv(os.path.join(path_raw, a))
        df_b = pd.read_csv(os.path.join(path_raw, b))
        try:
            assert(not(df_a.equals(df_b)))
        except AssertionError:
            print(f"{a} file is equal to {b} file")
            raise AssertionError

# def test_uniques_name_file():
#     all_files = listdir(SEPARATED_FILES_PATH)
#     unique_files = set(all_files)
#     assert(len(all_files)==len(unique_files))


if __name__ == "__main__":
    test_size_file()
    test_unique_date()
    test_rows()
    test_unique_content()
    # test_uniques_name_file()
