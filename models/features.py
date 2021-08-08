import os
import sys
import pandas as pd
from typing import Text

# TODO: code smell of project estructure
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
# parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

from src.config.config import Paths, Features

LABEL_KEY = "label"
df_train = pd.read_csv(f"{Paths.SPLITTED_DATA_PATH}train/train.csv")
FEATURE_KEYS = list(df_train.columns.values)
FEATURE_KEYS.remove(LABEL_KEY)

def transformed_name(key):
  return key + '_xf'
