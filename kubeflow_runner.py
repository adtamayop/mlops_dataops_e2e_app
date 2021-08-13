import os
from absl import logging
from tfx import v1 as tfx
from pipeline import configs
from pipeline import pipeline


OUTPUT_DIR = os.path.join('gs://', configs.GCS_BUCKET_NAME)

PIPELINE_ROOT = os.path.join(OUTPUT_DIR, 'tfx_pipeline_output',
                             configs.PIPELINE_NAME)

SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, 'serving_model')

DATA_PATH = 'gs://{}/data/processed/split/'.format(configs.GCS_BUCKET_NAME)


from pipeline import configs

def run():
  """Define a kubeflow pipeline."""

  # Metadata config. The defaults works work with the installation of
  # KF Pipelines using Kubeflow. If installing KF Pipelines using the
  # lightweight deployment option, you may need to override the defaults.
  # If you use Kubeflow, metadata will be written to MySQL database inside
  # Kubeflow cluster.
  metadata_config = tfx.orchestration.experimental.get_default_kubeflow_metadata_config(
  )

  runner_config = tfx.orchestration.experimental.KubeflowDagRunnerConfig(
      kubeflow_metadata_config=metadata_config,
      tfx_image=configs.PIPELINE_IMAGE)
  pod_labels = {
      'add-pod-env': 'true',
      tfx.orchestration.experimental.LABEL_KFP_SDK_ENV: 'tfx-template'
  }
  tfx.orchestration.experimental.KubeflowDagRunner(
      config=runner_config, pod_labels_to_attach=pod_labels
  ).run(
      pipeline.create_pipeline(
          pipeline_name=configs.PIPELINE_NAME,
          pipeline_root=PIPELINE_ROOT,
          data_path=DATA_PATH,
          # TODO(step 7): (Optional) Uncomment below to use BigQueryExampleGen.
          # query=configs.BIG_QUERY_QUERY,
          preprocessing_fn=configs.PREPROCESSING_FN,
          run_fn=configs.RUN_FN,
          train_args=tfx.proto.TrainArgs(num_steps=configs.TRAIN_NUM_STEPS),
          eval_args=tfx.proto.EvalArgs(num_steps=configs.EVAL_NUM_STEPS),
          eval_accuracy_threshold=configs.EVAL_ACCURACY_THRESHOLD,
          serving_model_dir=SERVING_MODEL_DIR,
          # TODO(step 7): (Optional) Uncomment below to use provide GCP related
          #               config for BigQuery with Beam DirectRunner.
          #beam_pipeline_args=configs
          # .BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS,
          # TODO(step 8): (Optional) Uncomment below to use Dataflow.
          # beam_pipeline_args=configs.DATAFLOW_BEAM_PIPELINE_ARGS,
          # TODO(step 9): (Optional) Uncomment below to use Cloud AI Platform.
          # ai_platform_training_args=configs.GCP_AI_PLATFORM_TRAINING_ARGS,
          # TODO(step 9): (Optional) Uncomment below to use Cloud AI Platform.
          # ai_platform_serving_args=configs.GCP_AI_PLATFORM_SERVING_ARGS,
      ))


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  run()
