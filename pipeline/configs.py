import os

PIPELINE_NAME = 'mlops-dataops-pipeline'

try:
  import google.auth
  try:
    _, GOOGLE_CLOUD_PROJECT = google.auth.default()
  except google.auth.exceptions.DefaultCredentialsError:
    GOOGLE_CLOUD_PROJECT = 'tfx-mlops-dataops-project'
except ImportError:
  GOOGLE_CLOUD_PROJECT = 'tfx-mlops-dataops-project'

GCS_BUCKET_NAME = 'tfx-mlops-dataops-project-kubeflowpipelines-default'

GOOGLE_CLOUD_REGION = "us-central1-a"

ENDPOINT = 'https://33a28ea90c347185-dot-us-central1.pipelines.googleusercontent.com'

# Following image will be used to run pipeline components run if Kubeflow Pipelines used.
# This image will be automatically built by CLI if we use --build-image flag.
PIPELINE_IMAGE = f'gcr.io/{GOOGLE_CLOUD_PROJECT}/{PIPELINE_NAME}'

PREPROCESSING_FN = 'models.preprocessing.preprocessing_fn'
RUN_FN = 'models.model.run_fn'

TRAIN_NUM_STEPS = 100
EVAL_NUM_STEPS = 15
VAL_NUM_STEPS = 15
EVAL_ACCURACY_THRESHOLD = 0.4
