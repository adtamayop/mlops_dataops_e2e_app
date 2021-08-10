import os
import sys
from typing import Text

import pandas as pd

# TODO: code smell of project estructure
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
# parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

from src.config.config import Features, Paths

LABEL_KEY = "label"
df_train = pd.read_csv(f"{Paths.DATA_TEST_PATH_FILE}")
FEATURE_KEYS = list(df_train.columns.values)
FEATURE_KEYS.remove(LABEL_KEY)

def transformed_name(key):
  return key + '_xf'
