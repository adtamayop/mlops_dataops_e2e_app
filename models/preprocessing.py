import tensorflow_transform as tft

from models import features


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
  for key in features.FEATURE_KEYS:
    outputs[features.transformed_name(key)] =  tft.scale_to_z_score(inputs[key])

  # Do not apply label transformation as it will result in wrong evaluation.
  outputs[features.transformed_name(
      features.LABEL_KEY)] = inputs[features.LABEL_KEY]

  return outputs
