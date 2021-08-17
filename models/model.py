import argparse
import os
import shutil
import sys
import tempfile
from typing import List, Text

import mlflow
import mlflow.tensorflow
import pandas as pd
import tensorflow as tf
import tensorflow_transform as tft
from absl import logging
from mlflow import pyfunc
from tensorflow import keras
from tensorflow.python.keras.models import Model
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform.tf_metadata import schema_utils
from tfx import v1 as tfx
from tfx_bsl.public import tfxio

# TODO: Modify project structure for don't do this smell code
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from models.preprocessing import transformed_name
from src.config.config import (DownloadDataParams, Features, LabellingParams,
                               ModelConstants)


def _get_serve_tf_examples_fn(model, schema, tf_transform_output):
    """Returns a function that parses a serialized tf.Example."""
    if tf_transform_output is None:  # Transform component is not used.

        @tf.function
        def serve_tf_examples_fn(serialized_tf_examples):
            """Returns the output to be used in the serving signature."""
            feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec
            feature_spec.pop(Features.LABEL_KEY)
            parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
            return model(parsed_features)

    else:  # Transform component exists.
        model.tft_layer = tf_transform_output.transform_features_layer()

        @tf.function
        def serve_tf_examples_fn(serialized_tf_examples):
            """Returns the output to be used in the serving signature."""
            feature_spec = tf_transform_output.raw_feature_spec()
            feature_spec.pop(Features.LABEL_KEY)
            parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
            transformed_features = model.tft_layer(parsed_features)
            return model(transformed_features)

    return serve_tf_examples_fn


def _input_fn(
    file_pattern: List[Text],
    data_accessor: tfx.components.DataAccessor,
    schema: schema_pb2.Schema,
    label: Text,
    batch_size: int = 200,
) -> tf.data.Dataset:
    """Generates features and label for tuning/training.

    Args:
      file_pattern: List of paths or patterns of input tfrecord files.
      data_accessor: DataAccessor for converting input to RecordBatch.
      schema: A schema proto of input data.
      label: Name of the label.
      batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch

    Returns:
      A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size, label_key=label),
        schema,
    ).repeat()


def _build_keras_model(feature_list: List[Text]) -> tf.keras.Model:
    """Creates a DNN Keras model for classifying

    Args:
      feature_list: List of feature names.

    Returns:
      A Keras Model.
    """
    inputs = [keras.layers.Input(shape=(1,), name=f) for f in feature_list]
    d = keras.layers.concatenate(inputs)
    for _ in range(ModelConstants.NUM_LAYERS):
        d = keras.layers.Dense(ModelConstants.HIDDEN_LAYER_UNITS, activation="relu")(d)
    outputs = keras.layers.Dense(
        ModelConstants.OUTPUT_LAYER_UNITS, activation="softmax"
    )(d)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(ModelConstants.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    model.summary(print_fn=logging.info)
    return model


def run_fn(fn_args: tfx.components.FnArgs):
    """Train the model based on given args.

    Args:
      fn_args: Holds args used to train the model as name/value pairs.
    """
    mlflow.tensorflow.autolog()

    with mlflow.start_run():

        mlflow.log_param("SYMBOL", DownloadDataParams.SYMBOL)
        mlflow.log_param("INTERVAL", DownloadDataParams.INTERVAL)
        mlflow.log_param("HIDDEN_LAYER_UNITS", ModelConstants.HIDDEN_LAYER_UNITS)
        mlflow.log_param("LEARNING_RATE", ModelConstants.LEARNING_RATE)
        mlflow.log_param("NUM_LAYERS", ModelConstants.NUM_LAYERS)
        mlflow.log_param("EVAL_BATCH_SIZE", ModelConstants.EVAL_BATCH_SIZE)
        mlflow.log_param("TRAIN_BATCH_SIZE", ModelConstants.TRAIN_BATCH_SIZE)
        mlflow.log_param("N_TSTEPS", LabellingParams.N_TSTEPS)
        mlflow.log_param(
            "ROLLING_AVG_WINDOW_SIZE", LabellingParams.ROLLING_AVG_WINDOW_SIZE
        )
        mlflow.log_param("STATIONARY_THRESHOLD", LabellingParams.STATIONARY_THRESHOLD)
        mlflow.log_param("RAW_INPUT_FEATURES", Features.RAW_INPUT_FEATURES)
        # mlflow.log_artifact(fn_args.serving_model_dir)

        if fn_args.transform_output is None:  # Transform is not used.
            tf_transform_output = None
            schema = tfx.utils.parse_pbtxt_file(
                fn_args.schema_file, schema_pb2.Schema()
            )
            feature_list = Features.FEATURE_KEYS
            label_key = Features.LABEL_KEY
        else:
            tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
            schema = tf_transform_output.transformed_metadata.schema
            feature_list = [transformed_name(f) for f in Features.FEATURE_KEYS]
            label_key = transformed_name(Features.LABEL_KEY)

        mirrored_strategy = tf.distribute.MirroredStrategy()
        train_batch_size = (
            ModelConstants.TRAIN_BATCH_SIZE * mirrored_strategy.num_replicas_in_sync
        )
        eval_batch_size = (
            ModelConstants.EVAL_BATCH_SIZE * mirrored_strategy.num_replicas_in_sync
        )

        train_dataset = _input_fn(
            fn_args.train_files,
            fn_args.data_accessor,
            schema,
            label_key,
            batch_size=train_batch_size,
        )
        eval_dataset = _input_fn(
            fn_args.eval_files,
            fn_args.data_accessor,
            schema,
            label_key,
            batch_size=eval_batch_size,
        )

        with mirrored_strategy.scope():
            model = _build_keras_model(feature_list)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=fn_args.model_run_dir, update_freq="batch"
        )

        model.fit(
            train_dataset,
            steps_per_epoch=fn_args.train_steps,
            validation_data=eval_dataset,
            validation_steps=fn_args.eval_steps,
            callbacks=[tensorboard_callback],
            epochs=ModelConstants.EPOCHS,
        )

        signatures = {
            "serving_default": _get_serve_tf_examples_fn(
                model, schema, tf_transform_output
            ).get_concrete_function(
                tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
            ),
        }
        model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)
