# Lint as: python3
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TFX penguin model features.

Define constants here that are common across all models
including features names, label and size of vocabulary.
"""
import pandas as pd
from typing import Text


import os
import sys


# TODO: code smell of project estructure
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
# parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)


# TODO(step 3, 4): Define constants for features of the model.
# FEATURE_KEYS = [
#     'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g'
# ]
# LABEL_KEY = 'species'


LABEL_KEY = "label"

df_train = pd.read_csv("data/data.csv")
FEATURE_KEYS = list(df_train.columns.values)
FEATURE_KEYS.remove(LABEL_KEY)

print(FEATURE_KEYS)

def transformed_name(key: Text) -> Text:
  """Generate the name of the transformed feature from original name."""
  return key + '_xf'
