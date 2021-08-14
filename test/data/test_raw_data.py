import filecmp
import itertools
import os
import sys
from os import listdir
from os.path import isfile, join

import pandas as pd

# TODO: code smell of project estructure
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

from src.config.config import Paths

SEPARATED_FILES_PATH = Paths.DATA_RAW_BY_MONTH
JOINNED_FILE_PATH = Paths.DATA_RAW_JOIN_PATH

def test_size_file():
    all_files = listdir(SEPARATED_FILES_PATH)
    for file in all_files:
        size = os.path.getsize(os.path.join(SEPARATED_FILES_PATH, file))
        assert(size>0)

def test_uniques_name_file():
    all_files = listdir(SEPARATED_FILES_PATH)
    unique_files = set(all_files)
    assert(len(all_files)==len(unique_files))

def test_unique_content():
    all_files = listdir(SEPARATED_FILES_PATH)
    couple_combination = list(itertools.combinations(all_files, 2))
    for a,b in couple_combination:
        df_a = pd.read_csv(os.path.join(SEPARATED_FILES_PATH, a))
        df_b = pd.read_csv(os.path.join(SEPARATED_FILES_PATH, b))
        try:
            assert(not(df_a.equals(df_b)))
        except AssertionError:
            print(f"{a} file is equal to {b} file")
            raise AssertionError



if __name__ == "__main__":
    test_size_file()
    test_uniques_name_file()
    test_unique_content()
