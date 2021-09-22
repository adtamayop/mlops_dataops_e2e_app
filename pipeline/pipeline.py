
import os
import sys
from typing import List, Optional, Text

import tensorflow as tf
import tensorflow_model_analysis as tfma
from ml_metadata.proto import metadata_store_pb2
from tfx import v1 as tfx
from tfx.proto import (bulk_inferrer_pb2, example_gen_pb2, pusher_pb2,
                       trainer_pb2, transform_pb2)

# from models import features
from pipeline.configs import EVAL_NUM_STEPS, TRAIN_NUM_STEPS, VAL_NUM_STEPS

# TODO: Modify project structure for don't do this smell code
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from src.config.config import Features


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_path: Text,
    preprocessing_fn: Text,
    run_fn: Text,
    train_args: tfx.proto.TrainArgs,
    eval_args: tfx.proto.EvalArgs,
    eval_accuracy_threshold: float,
    serving_model_dir: Text,
    metadata_connection_config: Optional[
        metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None,
) -> tfx.dsl.Pipeline:
  """Implements the pipeline with TFX."""

  components = []

  input = example_gen_pb2.Input(
        splits=[
            example_gen_pb2.Input.Split(name="train", pattern="train/*"),
            example_gen_pb2.Input.Split(name="validation", pattern="val/*"),
            example_gen_pb2.Input.Split(name="test", pattern="test/*"),
        ]
    )

  example_gen = tfx.components.CsvExampleGen(input_base=data_path, input_config=input)
  components.append(example_gen)

  statistics_gen = tfx.components.StatisticsGen(
      examples=example_gen.outputs['examples'])
  components.append(statistics_gen)

  schema_gen = tfx.components.SchemaGen(
      statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)
  components.append(schema_gen)

  example_validator = tfx.components.ExampleValidator(  # pylint: disable=unused-variable
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])
  components.append(example_validator)

  # Performs transformations and feature engineering in training and serving.
  transform = tfx.components.Transform(  # pylint: disable=unused-variable
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      preprocessing_fn=preprocessing_fn,
      splits_config=transform_pb2.SplitsConfig(
            analyze=["train"], transform=["train", "validation", "test"]
        ))
  # TODO(step 3): Uncomment here to add Transform to the pipeline.
  components.append(transform)

  # Uses user-provided Python function that implements a model using Tensorflow.
  trainer = tfx.components.Trainer(
      run_fn=run_fn,
      examples=transform.outputs['transformed_examples'],
      transform_graph=transform.outputs['transform_graph'],
      schema=schema_gen.outputs['schema'],
      train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=TRAIN_NUM_STEPS),
      eval_args=trainer_pb2.EvalArgs(splits=["validation"], num_steps=VAL_NUM_STEPS),
    )
  components.append(trainer)

  model_resolver = tfx.dsl.Resolver(
      strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
      model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
      model_blessing=tfx.dsl.Channel(
          type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
              'latest_blessed_model_resolver')
  components.append(model_resolver)


  metrics = [
        tf.keras.metrics.SparseCategoricalCrossentropy(name='sparse_categorical_crossentropy'),
        tf.keras.metrics.SparseCategoricalAccuracy(name="SparseCategoricalAccuracy"),
        tfma.metrics.MultiClassConfusionMatrixPlot(
            name='multi_class_confusion_matrix_plot'),
    ]

  metrics_specs = tfma.metrics.specs_from_metrics(metrics)


  eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(label_key=Features.LABEL_KEY)],
      slicing_specs=[tfma.SlicingSpec()],
      metrics_specs= metrics_specs

    #   [
    #       tfma.MetricsSpec(
    #           metrics=[

                # tfma.MetricConfig(
                #     class_name='SparseCategoricalAccuracy',
                #     threshold=tfma.MetricThreshold(
                #         value_threshold=tfma.GenericValueThreshold(
                #             lower_bound={'value': eval_accuracy_threshold}),
                #         change_threshold=tfma.GenericChangeThreshold(
                #             direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                #             absolute={'value': -1e-10}))
                # ),
                # tfma.metrics.ConfusionMatrixPlot(
                #     name = "CONFUSION_MATRIX_PLOT_NAME"
                # ),
                # tfma.metrics.F1Score(
                #     name = "F1_SCORE_NAME"
                # )
        #   ])
    #   ]
      )
  evaluator = tfx.components.Evaluator(  # pylint: disable=unused-variable
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      eval_config=eval_config,
      example_splits=["test"])
  components.append(evaluator)

  pusher = tfx.components.Pusher(  # pylint: disable=unused-variable
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      push_destination=tfx.proto.PushDestination(
          filesystem=tfx.proto.PushDestination.Filesystem(
              base_directory=serving_model_dir)))
  components.append(pusher)

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components,
      enable_cache=True,
      metadata_connection_config=metadata_connection_config,
      beam_pipeline_args=beam_pipeline_args,
  )
