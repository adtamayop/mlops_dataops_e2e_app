import os
import sys
from re import X
from typing import List

import absl
import kerastuner as keras_tuner
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import tensorflow_transform as tft
from absl import logging
from metrics import f1_metric, f1_weighted
from tensorflow import keras
from tensorflow.keras.utils import get_custom_objects
from tensorflow.python.keras import constraints
from tensorflow.python.keras.backend import constant
from tfx import v1 as tfx

# TODO: Modify project structure for don't do this smell code
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from tensorflow.keras import datasets, layers, models
from tfx_bsl.public import tfxio

from config.config import Features
from preprocessing import transformed_name

TRAIN_BATCH_SIZE = 20
EVAL_BATCH_SIZE = 10

def make_serving_signatures(model,
                            tf_transform_output: tft.TFTransformOutput):
  """Returns the serving signatures.

  Args:
    model: the model function to apply to the transformed features.
    tf_transform_output: The transformation to apply to the serialized
      tf.Example.

  Returns:
    The signatures to use for saving the mode. The 'serving_default' signature
    will be a concrete function that takes a batch of unspecified length of
    serialized tf.Example, parses them, transformes the features and
    then applies the model. The 'transform_features' signature will parses the
    example and transforms the features.
  """

  # We need to track the layers in the model in order to save it.
  # TODO(b/162357359): Revise once the bug is resolved.
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
  ])
  def serve_tf_examples_fn(serialized_tf_example):
    """Returns the output to be used in the serving signature."""
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    # Remove label feature since these will not be present at serving time.
    raw_feature_spec.pop(Features.LABEL_KEY)
    raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
    transformed_features = model.tft_layer(raw_features)
    logging.info('serve_transformed_features = %s', transformed_features)

    outputs = model(transformed_features)
    # TODO(b/154085620): Convert the predicted labels from the model using a
    # reverse-lookup (opposite of transform.py).
    return {'outputs': outputs}

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
  ])
  def transform_features_fn(serialized_tf_example):
    """Returns the transformed_features to be fed as input to evaluator."""
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
    transformed_features = model.tft_layer(raw_features)
    logging.info('eval_transformed_features = %s', transformed_features)
    return transformed_features

  return {
      'serving_default': serve_tf_examples_fn,
      'transform_features': transform_features_fn
  }


def input_fn(file_pattern: List[str],
             data_accessor: tfx.components.DataAccessor,
             tf_transform_output: tft.TFTransformOutput,
             batch_size: int) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  return data_accessor.tf_dataset_factory(
      file_pattern,
      tfxio.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=transformed_name(Features.LABEL_KEY)),
      tf_transform_output.transformed_metadata.schema).repeat()


def _get_hyperparameters() -> keras_tuner.HyperParameters:
  """Returns hyperparameters for building Keras model."""
  hp = keras_tuner.HyperParameters()
  # Defines search space.
  hp.Choice('learning_rate', [0.01, 0.001], default= 0.001)
  hp.Choice('conv2d_layer_1',[16,32,64], default=32)
  hp.Choice('conv2d_kernel_size_1',[2, 3], default=3)
  hp.Choice('conv2d_strides_1',[1, 2], default=1)
  hp.Choice('activation_layer_1',["relu", "sigmoid"], default="relu")
  hp.Choice('dropout', [0.1, 0.2, 0.3], default=0.2)
  hp.Choice('dense_layer_2', [16,32,64], default=32)
  hp.Choice('dense_layers', [1,2,4], default=2)
  hp.Choice('dense_layer_n', [16,32,64], default=32)
  return hp




def _make_keras_model(hparams: keras_tuner.HyperParameters) -> tf.keras.Model:
  """Creates a DNN Keras model for classifying penguin data.

  Args:
    hparams: Holds HyperParameters for tuning.

  Returns:
    A Keras Model.
  """

  get_custom_objects().update({"f1_metric": f1_metric, "f1_weighted": f1_weighted})

  inputs = [
      keras.layers.Input(shape=(1,), name=transformed_name(f))
      for f in Features.FEATURE_KEYS
  ]
  d = keras.layers.concatenate(inputs)
  d = tf.keras.layers.Reshape((15,15, 1))(d)
  d = tf.keras.layers.Conv2D(
    64, 3, strides=1,padding='same',
    activation="relu", use_bias=True,
    kernel_initializer='glorot_uniform')(d)

  d = tf.keras.layers.Dropout(hparams.get("dropout"))(d)


  d = tf.keras.layers.Flatten()(d)
  d = keras.layers.Dense(hparams.get("dense_layer_2"), activation='relu')(d)

  for _ in range(hparams.get("dense_layers")):
        d = keras.layers.Dense(
          hparams.get("dense_layer_n"), activation="relu")(d)

  outputs = keras.layers.Dense(3, activation='softmax')(d)

  model = keras.Model(inputs=inputs, outputs=outputs)



  optimizer=keras.optimizers.Adam(hparams.get('learning_rate'))

  model.compile(
      optimizer=optimizer,
      loss='sparse_categorical_crossentropy',
      metrics=[keras.metrics.SparseCategoricalAccuracy()
              # tf.keras.metrics.Precision()
              ])

  model.summary(print_fn=absl.logging.info)
  return model

def tuner_fn(fn_args: tfx.components.FnArgs) -> tfx.components.TunerFnResult:
  """Build the tuner using the KerasTuner API.

  Args:
    fn_args: Holds args as name/value pairs.
      - working_dir: working dir for tuning.
      - train_files: List of file paths containing training tf.Example data.
      - eval_files: List of file paths containing eval tf.Example data.
      - train_steps: number of train steps.
      - eval_steps: number of eval steps.
      - schema_path: optional schema of the input data.
      - transform_graph_path: optional transform graph produced by TFT.

  Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
  """
  tuner = keras_tuner.RandomSearch(
      _make_keras_model,
      max_trials=1,
      hyperparameters=_get_hyperparameters(),
      allow_new_entries=False,
      objective=keras_tuner.Objective('val_sparse_categorical_accuracy', 'max'),
      directory=fn_args.working_dir,
      project_name='pipeline_tuning')

  transform_graph = tft.TFTransformOutput(fn_args.transform_graph_path)

  train_dataset = input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      transform_graph,
      TRAIN_BATCH_SIZE)

  eval_dataset = input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      transform_graph,
      EVAL_BATCH_SIZE)

  return tfx.components.TunerFnResult(
      tuner=tuner,
      fit_kwargs={
          'x': train_dataset,
          'validation_data': eval_dataset,
          'steps_per_epoch': fn_args.train_steps,
          'validation_steps': fn_args.eval_steps
      })


def run_fn(fn_args: tfx.components.FnArgs):

  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      tf_transform_output,
      TRAIN_BATCH_SIZE)

  eval_dataset = input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      tf_transform_output,
      EVAL_BATCH_SIZE)

  if fn_args.hyperparameters:
    hparams = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)
  else:
    # This is a shown case when hyperparameters is decided and Tuner is removed
    # from the pipeline. User can also inline the hyperparameters directly in
    # _build_keras_model.
    hparams = _get_hyperparameters()
  absl.logging.info('HyperParameters for training: %s' % hparams.get_config())

  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = _make_keras_model(hparams)

  mlflow.tensorflow.autolog()

  with mlflow.start_run():
      mlflow.log_param("learning_rate", hparams.get('learning_rate'))
      mlflow.log_param("Dense_1 units", hparams.get('dense_layer_2'))
      mlflow.log_param("conv2d_layer_1 units", hparams.get('conv2d_layer_1'))
      mlflow.log_param("conv2d_kernel_size_1", hparams.get('conv2d_kernel_size_1'))
      mlflow.log_param("conv2d_strides_1", hparams.get('conv2d_strides_1'))
      mlflow.log_param("activation_layer_1", hparams.get('activation_layer_1'))
      mlflow.log_param("dropout", hparams.get('dropout'))
      mlflow.log_param("dense_layer_2 units", hparams.get('dense_layer_2'))
      mlflow.log_param("dense_layers aditionals", hparams.get('dense_layers'))
      # mlflow.log_artifact(fn_args.serving_model_dir)

      # Write logs to path
      tensorboard_callback = tf.keras.callbacks.TensorBoard(
          log_dir=fn_args.model_run_dir, update_freq='batch')


      es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                    patience=100, min_delta=0.0001)

      rlp = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.02, patience=20, verbose=1, mode='min',
                              min_delta=0.001, cooldown=1, min_lr=0.0001)



      model.fit(
          train_dataset,
          epochs = 10,
          steps_per_epoch=fn_args.train_steps,
          validation_data=eval_dataset,
          validation_steps=fn_args.eval_steps,
          callbacks=[tensorboard_callback, es, rlp])

      signatures = make_serving_signatures(model, tf_transform_output)
      model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
