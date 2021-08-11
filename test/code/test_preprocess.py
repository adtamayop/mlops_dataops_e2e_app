
import os
import sys

# TODO: code smell of project estructure
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

from models.preprocessing import transformed_name


def test_features_names():
    key = 'fare'
    xfm_key = transformed_name(key)
    assert (xfm_key, 'fare_xf')
