import multiprocessing
import os
import socket
import sys
from typing import List, Optional

import absl
import tensorflow as tf
import tensorflow_model_analysis as tfma
from absl import flags
from tfx import v1 as tfx
from tfx.dsl.experimental.conditionals import conditional
from tfx.proto import bulk_inferrer_pb2, example_gen_pb2, transform_pb2

from src.config.config import Features
from src.preprocessing.preprocessing import transformed_name

_penguin_root = os.path.join(".")

# _data_root = os.path.join(_penguin_root, 'data/train_data/WMT/train/')

_tfx_root = os.path.join(_penguin_root, 'tfx_pipeline_output')


def _create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_root: str,
    module_file: str,
    preprocess_file: str,
    accuracy_threshold: float,
    serving_model_dir: str,
    metadata_path: str,
    user_provided_schema_path: Optional[str],
    enable_tuning: bool,
    enable_bulk_inferrer: bool,
    enable_transform_input_cache: bool
) -> tfx.dsl.Pipeline:
  """Implements the penguin pipeline with TFX.

  Args:
    pipeline_name: name of the TFX pipeline being created.
    pipeline_root: root directory of the pipeline.
    data_root: directory containing the penguin data.
    module_file: path to files used in Trainer and Transform components.
    accuracy_threshold: minimum accuracy to push the model.
    serving_model_dir: filepath to write pipeline SavedModel to.
    metadata_path: path to local pipeline ML Metadata store.
    user_provided_schema_path: path to user provided schema file.
    enable_tuning: If True, the hyperparameter tuning through KerasTuner is
      enabled.
    enable_bulk_inferrer: If True, the generated model will be used for a
      batch inference.
    examplegen_input_config: ExampleGen's input_config.
    examplegen_range_config: ExampleGen's range_config.
    resolver_range_config: SpansResolver's range_config. Specify this will
      enable SpansResolver to get a window of ExampleGen's output Spans for
      transform and training.
    beam_pipeline_args: list of beam pipeline options for LocalDAGRunner. Please
      refer to https://beam.apache.org/documentation/runners/direct/.
    enable_transform_input_cache: Indicates whether input cache should be used
      in Transform if available.

  Returns:
    A TFX pipeline object.
  """
  input = example_gen_pb2.Input(
        splits=[
            example_gen_pb2.Input.Split(name="train", pattern="train/*"),
            example_gen_pb2.Input.Split(name="validation", pattern="val/*"),
            example_gen_pb2.Input.Split(name="test", pattern="test/*"),
        ]
    )

  example_gen = tfx.components.CsvExampleGen(input_base=data_root, input_config=input)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = tfx.components.StatisticsGen(
      examples=example_gen.outputs['examples'])

  if user_provided_schema_path:
    # Import user-provided schema.
    schema_importer = tfx.dsl.Importer(
        source_uri=user_provided_schema_path,
        artifact_type=tfx.types.standard_artifacts.Schema).with_id(
            'schema_importer')
    schema = schema_importer.outputs['result']
  else:
    # Generates schema based on statistics files.
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True)
    schema = schema_gen.outputs['schema']

  # Performs anomaly detection based on statistics and data schema.
  example_validator = tfx.components.ExampleValidator(
      statistics=statistics_gen.outputs['statistics'], schema=schema)

  # Gets multiple Spans for transform and training.
  # if resolver_range_config:
  #   examples_resolver = tfx.dsl.Resolver(
  #       strategy_class=tfx.dsl.experimental.SpanRangeStrategy,
  #       config={
  #           'range_config': resolver_range_config
  #       },
  #       examples=tfx.dsl.Channel(
  #           type=tfx.types.standard_artifacts.Examples,
  #           producer_component_id=example_gen.id)).with_id('span_resolver')

  # Performs transformations and feature engineering in training and serving.
  if enable_transform_input_cache:
    transform_cache_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestArtifactStrategy,
        cache=tfx.dsl.Channel(type=tfx.types.standard_artifacts.TransformCache)
    ).with_id('transform_cache_resolver')
    tft_resolved_cache = transform_cache_resolver.outputs['cache']
  else:
    tft_resolved_cache = None

  transform = tfx.components.Transform(
      examples=example_gen.outputs['examples'],
      schema=schema,
      module_file=preprocess_file,
          splits_config=transform_pb2.SplitsConfig(
          analyze=["train"], transform=["train", "validation", "test"]),
      analyzer_cache=tft_resolved_cache)

  # Tunes the hyperparameters for model training based on user-provided Python
  # function. Note that once the hyperparameters are tuned, you can drop the
  # Tuner component from pipeline and feed Trainer with tuned hyperparameters.
  enable_tuning=False
  if enable_tuning:
    tuner = tfx.components.Tuner(
        module_file=module_file,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=tfx.proto.TrainArgs(splits=["train"], num_steps=20),
        eval_args=tfx.proto.EvalArgs(splits=["validation"], num_steps=5))

  # Uses user-provided Python function that trains a model.
  trainer = tfx.components.Trainer(
      module_file=module_file,
      examples=transform.outputs['transformed_examples'],
      transform_graph=transform.outputs['transform_graph'],
      schema=schema,
      # If Tuner is in the pipeline, Trainer can take Tuner's output
      # best_hyperparameters artifact as input and utilize it in the user module
      # code.
      #
      # If there isn't Tuner in the pipeline, either use ImporterNode to import
      # a previous Tuner's output to feed to Trainer, or directly use the tuned
      # hyperparameters in user module code and set hyperparameters to None
      # here.
      #
      # Example of ImporterNode,
      #   hparams_importer = ImporterNode(
      #     source_uri='path/to/best_hyperparameters.txt',
      #     artifact_type=HyperParameters).with_id('import_hparams')
      #   ...
      #   hyperparameters = hparams_importer.outputs['result'],
      hyperparameters=(tuner.outputs['best_hyperparameters']
                       if enable_tuning else None),
      train_args=tfx.proto.TrainArgs(splits=["train"], num_steps=100),
      eval_args=tfx.proto.EvalArgs(splits=["validation"], num_steps=5),
      )

  # Get the latest blessed model for model validation.
  model_resolver = tfx.dsl.Resolver(
      strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
      model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
      model_blessing=tfx.dsl.Channel(
          type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
              'latest_blessed_model_resolver')

  metrics = [
        tfma.MetricConfig(
        class_name='SparseCategoricalAccuracy',
        threshold=tfma.MetricThreshold(
            value_threshold=tfma.GenericValueThreshold(
                lower_bound={'value': accuracy_threshold}),
            # Change threshold will be ignored if there is no
            # baseline model resolved from MLMD (first run).
            change_threshold=tfma.GenericChangeThreshold(
                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                absolute={'value': -1e-10}))),
        tfma.metrics.MultiClassConfusionMatrixPlot(
            name='multi_class_confusion_matrix_plot'),
        tfma.metrics.F1Score(
            name='F1Score'),
    ]

  # metrics_specs = tfma.metrics.specs_from_metrics(metrics)
  # Uses TFMA to compute evaluation statistics over features of a model and
  # perform quality validation of a candidate model (compared to a baseline).
  eval_config = tfma.EvalConfig(
      model_specs=[
          tfma.ModelSpec(
              signature_name='serving_default',
              label_key=transformed_name(Features.LABEL_KEY),
              preprocessing_function_names=['transform_features'])
      ],
      slicing_specs=[tfma.SlicingSpec()],
      metrics_specs=[
        tfma.MetricsSpec(
            metrics=[
              tfma.MetricConfig(class_name='ExampleCount'),
              tfma.MetricConfig(class_name="MultiClassConfusionMatrixPlot"),
              tfma.MetricConfig(class_name='F1Score'),
              tfma.MetricConfig(
                  class_name='SparseCategoricalAccuracy',
                  threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': accuracy_threshold}),
                      # Change threshold will be ignored if there is no
                      # baseline model resolved from MLMD (first run).
                      change_threshold=tfma.GenericChangeThreshold(
                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                          absolute={'value': -1e-10}))),

            ]
          ),
        ]
      )


  evaluator = tfx.components.Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      eval_config=eval_config,
      example_splits=["test"])

  # make_warmup = True

  #   # Performs infra validation of a candidate model to prevent unservable model
  # # from being pushed. This config will launch a model server of the latest
  # # TensorFlow Serving image in a local docker engine.
  # infra_validator = tfx.components.InfraValidator(
  #     model=trainer.outputs['model'],
  #     examples=example_gen.outputs['examples'],
  #     serving_spec=tfx.proto.ServingSpec(
  #         tensorflow_serving=tfx.proto.TensorFlowServing(tags=['latest']),
  #         local_docker=tfx.proto.LocalDockerConfig()),
  #     request_spec=tfx.proto.RequestSpec(
  #         tensorflow_serving=tfx.proto.TensorFlowServingRequestSpec(),
  #         # If this flag is set, InfraValidator will produce a model with
  #         # warmup requests (in its outputs['blessing']).
  #         make_warmup=make_warmup))

  # # Checks whether the model passed the validation steps and pushes the model
  # # to a file destination if check passed.

  # # Checks whether the model passed the validation steps and pushes the model
  # # to a file destination if check passed.
  # if make_warmup:
  #   # If InfraValidator.request_spec.make_warmup = True, its output contains
  #   # a model so that Pusher can push 'infra_blessing' input instead of
  #   # 'model' input.
  #   pusher = tfx.components.Pusher(
  #       model_blessing=evaluator.outputs['blessing'],
  #       infra_blessing=infra_validator.outputs['blessing'],
  #       push_destination=tfx.proto.PushDestination(
  #           filesystem=tfx.proto.PushDestination.Filesystem(
  #               base_directory=serving_model_dir)))
  # else:
  #   # Otherwise, 'infra_blessing' does not contain a model and is used as a
  #   # conditional checker just like 'model_blessing' does. This is the typical
  #   # use case.
  #   pusher = tfx.components.Pusher(
  #       model=trainer.outputs['model'],
  #       model_blessing=evaluator.outputs['blessing'],
  #       infra_blessing=infra_validator.outputs['blessing'],
  #       push_destination=tfx.proto.PushDestination(
  #           filesystem=tfx.proto.PushDestination.Filesystem(
  #               base_directory=serving_model_dir)))


  # Components declared within the conditional block will only be triggered
  # if the Predicate evaluates to True.
  #
  # In the example below,
  # evaluator.outputs['blessing'].future()[0].custom_property('blessed') == 1
  # is a Predicate, which will be evaluated during runtime.
  #
  # - evaluator.outputs['blessing'] is the output Channel 'blessing'.
  # - .future() turns the Channel into a Placeholder.
  # - [0] gets the first artifact from the 'blessing' Channel.
  # - .custom_property('blessed') gets a custom property called 'blessed' from
  #   that artifact.
  # - == 1 compares that property with 1. (An explicit comparison is needed.
  #   There's no automatic boolean conversion based on truthiness.)
  #
  # Note these operations are just placeholder, something like Mocks. They are
  # not evaluated until runtime. For more details, see tfx/dsl/placeholder/.



  with conditional.Cond(evaluator.outputs['blessing'].future()
                        [0].custom_property('blessed') == 1):
    # Checks whether the model passed the validation steps and pushes the model
    # to a file destination if check passed.
    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        # No need to pass model_blessing any more, since Pusher is already
        # guarded by a Conditional.
        # model_blessing=evaluator.outputs['blessing'],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir)))


  # # Showcase for BulkInferrer component.
  # if enable_bulk_inferrer:
  #   # Generates unlabelled examples.
  #   example_gen_unlabelled = tfx.components.CsvExampleGen(
  #       input_base=os.path.join('data_bulk/')).with_id(
  #           'CsvExampleGen_Unlabelled')

  #   # Performs offline batch inference.
  #   bulk_inferrer = tfx.components.BulkInferrer(
  #       examples=example_gen_unlabelled.outputs['examples'],
  #       model=trainer.outputs['model'],
  #       # Empty data_spec.example_splits will result in using all splits.
  #       data_spec=tfx.proto.DataSpec(),
  #       model_spec=tfx.proto.ModelSpec()
  #    )

  components_list = [
      example_gen,
      statistics_gen,
      example_validator,
      transform,
      trainer,
      model_resolver,
      evaluator,
      # infra_validator,
      pusher,
  ]
  if user_provided_schema_path:
    components_list.append(schema_importer)
  else:
    components_list.append(schema_gen)
  # if resolver_range_config:
  #   components_list.append(examples_resolver)
  if enable_transform_input_cache:
    components_list.append(transform_cache_resolver)
  if enable_tuning:
    components_list.append(tuner)
  # if enable_bulk_inferrer:
  #   components_list.append(example_gen_unlabelled)
  #   components_list.append(bulk_inferrer)

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components_list,
      enable_cache=True,
      metadata_connection_config=tfx.orchestration.metadata
      .sqlite_metadata_connection_config(metadata_path),
      )
