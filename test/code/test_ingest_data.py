
import json
import math
import os
import sys

# TODO: code smell of project estructure
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

from src.config.config import DownloadDataParams, Features


def test_type_donwload_data():
    type_data = DownloadDataParams.DATA_TYPE
    assert(type_data=="csv" or type_data=="json")

def test_n_features_perfect_root():
    assert(math.sqrt(Features.N_FEATURES).is_integer())

def test_minimum_img_size():
    assert(math.sqrt(Features.N_FEATURES)>=10)


if __name__ == "__main__":
    # test_months_to_download()
    # test_years_to_download()
    test_type_donwload_data()
    test_n_features_perfect_root()
    test_minimum_img_size()
