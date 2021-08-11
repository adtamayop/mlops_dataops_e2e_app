import os
import sys

import tensorflow_transform as tft

# from models import features

# TODO: Modify project structure for don't do this smell code
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from src.config.config import Features


# TFX Transform will call this function.
# TODO(step 3): Define your transform logic in this function.
def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}

  # This function is the entry point for your feature engineering with
  # TensorFlow Transform, using the TFX Transform component.  In this example
  # the feature engineering is very simple, only applying z-score scaling.
  for key in Features.FEATURE_KEYS:
    outputs[transformed_name(key)] =  tft.scale_to_z_score(inputs[key])

  # Do not apply label transformation as it will result in wrong evaluation.
  outputs[transformed_name(
      Features.LABEL_KEY)] = inputs[Features.LABEL_KEY]

  return outputs


def transformed_name(key):
  return key + '_xf'
