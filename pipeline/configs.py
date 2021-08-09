import os  # pylint: disable=unused-import

# TODO(b/149347293): Move more TFX CLI flags into python configuration.

# Pipeline name will be used to identify this pipeline.
PIPELINE_NAME = 'pipeline_tfx'

# GCP related configs.

# Following code will retrieve your GCP project. You can choose which project
# to use by setting GOOGLE_CLOUD_PROJECT environment variable.
try:
  import google.auth  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
  try:
    _, GOOGLE_CLOUD_PROJECT = google.auth.default()
  except google.auth.exceptions.DefaultCredentialsError:
    GOOGLE_CLOUD_PROJECT = 'tfx-pipeline'
except ImportError:
  GOOGLE_CLOUD_PROJECT = 'tfx-pipeline'


GCS_BUCKET_NAME = 'main-cyclist-321921-kubeflowpipelines-default'


PIPELINE_IMAGE = f'gcr.io/{GOOGLE_CLOUD_PROJECT}/{PIPELINE_NAME}'

BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS = [
'--project=' + GOOGLE_CLOUD_PROJECT,
'--temp_location=' + os.path.join('gs://', GCS_BUCKET_NAME, 'tmp'),
]

# Following image will be used to run pipeline components run if Kubeflow
# Pipelines used.
# This image will be automatically built by CLI if we use --build-image flag.
PIPELINE_IMAGE = f'gcr.io/{GOOGLE_CLOUD_PROJECT}/{PIPELINE_NAME}'

PREPROCESSING_FN = 'models.preprocessing.preprocessing_fn'
RUN_FN = 'models.model.run_fn'

TRAIN_NUM_STEPS = 100
EVAL_NUM_STEPS = 15
VAL_NUM_STEPS = 15

# Change this value according to your use cases.
EVAL_ACCURACY_THRESHOLD = 0.4
