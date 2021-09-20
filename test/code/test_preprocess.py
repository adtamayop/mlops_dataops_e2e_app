import os
import sys

# TODO: code smell of project estructure
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

from src.preprocessing.preprocessing import transformed_name


def test_features_names():
    key = 'test_feature'
    xfm_key = transformed_name(key)
    assert (xfm_key, 'test_feature_xf')

# TODO: scaler test
