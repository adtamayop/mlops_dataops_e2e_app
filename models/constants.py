"""Constants for the penguin model.

These values can be tweaked to affect model training performance.
"""

# Defines constants for the model. These constants can be determined via
# experiments using TFX Tuner component.
HIDDEN_LAYER_UNITS = 25
OUTPUT_LAYER_UNITS = 3
NUM_LAYERS = 5
LEARNING_RATE = 0.001

TRAIN_BATCH_SIZE = 20
EVAL_BATCH_SIZE = 10
