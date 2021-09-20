import os
import sys

import tensorflow_transform as tft

# TODO: Modify project structure for don't do this smell code
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from config.config import Features


# TFX Transform will call this function.
def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}

  for key in Features.FEATURE_KEYS:
    # tft.scale_to_z_score computes the mean and variance of the given feature
    # and scales the output based on the result.


    # outputs[transformed_name(key)] = tft.scale_to_z_score(inputs[key])

    outputs[transformed_name(key)] = inputs[key]


  # TODO(b/157064428): Support label transformation for Keras.
  # Do not apply label transformation as it will result in wrong evaluation.
  outputs[transformed_name(Features.LABEL_KEY)] = inputs[Features.LABEL_KEY]

  return outputs


# TFX Transform will call this function.
def transformed_name(key):
  return key + '_xf'
