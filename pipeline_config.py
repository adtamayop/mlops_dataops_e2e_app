import os

import pandas as pd

model = "time_series"
EXECUTION_PIPELINE = "local"

if EXECUTION_PIPELINE == "airflow":
    project_root = os.environ.get("project_dir")
    PIPELINE_BASE_DIR = f"{project_root}/tfx_pipeline/airflow_execution"
elif EXECUTION_PIPELINE == "local":
    project_root = "."
    PIPELINE_BASE_DIR = f"{project_root}/tfx_pipeline/local_execution"

if model == "time_series":
    # PIPELINE_BASE_DIR = f"{project_root}/tfx_pipeline"
    PIPELINE_NAME = "pipeline_tfx"
    SCHEMA_PIPELINE_NAME = "pipeline_tfx_schema"

    PIPELINE_ROOT = os.path.join(PIPELINE_BASE_DIR, PIPELINE_NAME)
    SCHEMA_PIPELINE_ROOT = os.path.join(PIPELINE_BASE_DIR, SCHEMA_PIPELINE_NAME)

    SCHEMA_METADATA_PATH = os.path.join(
        PIPELINE_BASE_DIR, "metadata", SCHEMA_PIPELINE_NAME, "metadata.db"
    )
    METADATA_PATH = os.path.join(
        PIPELINE_BASE_DIR, "metadata", PIPELINE_NAME, "metadata.db"
    )

    SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, "serving_model", PIPELINE_NAME)

    ENABLE_CACHE = False


    TRAIN_NUM_STEPS = 100
    EVAL_NUM_STEPS = 15


    try:
        import google.auth  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
        try:
            _, GOOGLE_CLOUD_PROJECT = google.auth.default()
        except google.auth.exceptions.DefaultCredentialsError:
            GOOGLE_CLOUD_PROJECT = 'tfx-pipeline'
    except ImportError:
        GOOGLE_CLOUD_PROJECT = 'tfx-pipeline'

    # Specify your GCS bucket name here. You have to use GCS to store output files
    # when running a pipeline with Kubeflow Pipeline on GCP or when running a job
    # using Dataflow. Default is '<gcp_project_name>-kubeflowpipelines-default'.
    # This bucket is created automatically when you deploy KFP from marketplace.
    GCS_BUCKET_NAME = 'main-cyclist-321921-kubeflowpipelines-default'

    # Following image will be used to run pipeline components run if Kubeflow
    # Pipelines used.
    # This image will be automatically built by CLI if we use --build-image flag.
    PIPELINE_IMAGE = f'gcr.io/{GOOGLE_CLOUD_PROJECT}/{PIPELINE_NAME}'


    # Change this value according to your use cases.
    EVAL_ACCURACY_THRESHOLD = 0.4

    BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS = [
    '--project=' + GOOGLE_CLOUD_PROJECT,
    '--temp_location=' + os.path.join('gs://', GCS_BUCKET_NAME, 'tmp'),
    ]


    # DATA_ROOT = f"{project_root}/data/raw/"
    DATA_ROOT = f"{project_root}/data/new_method/"
    DATA_BULK_ROOT = f"{project_root}/data/raw/bulk_infer/"
    PREPROCESSING_FN = f"{project_root}/src/pipeline/preprocessing.py"
    TRAINER_MODULE_FILE = f"{project_root}/src/pipeline/train.py"

    # LABELS and FEATURES
    # LABEL_KEY = "close_t"
    LABEL_KEY = "label"

    df_train = pd.read_csv(f"{DATA_ROOT}train_data/train_data.csv")
    FEATURE_KEYS = list(df_train.columns.values)
    FEATURE_KEYS.remove(LABEL_KEY)

    # BATCH_SIZE and STEPS
    TRAIN_BATCH_SIZE = 64
    VAL_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32
    TRAIN_STEPS = 2
    VAL_STEPS = 1

    # METRICS
    EVAL_ACCURACY_THRESHOLD = 0.0001

    # TEST
    TEST_PIPELINE_PATH = f"{project_root}/tests/test_pipeline/"
    TEST_COMPONENTS_PATH = f"{project_root}/tests/test_components/"

elif model == "penguin":

    PIPELINE_BASE_DIR = f"{project_root}/penguin_pipeline"

    PIPELINE_NAME = "pipeline_penguin"
    SCHEMA_PIPELINE_NAME = "pipeline_tfx_schema"

    PIPELINE_ROOT = os.path.join(PIPELINE_BASE_DIR, PIPELINE_NAME)
    SCHEMA_PIPELINE_ROOT = os.path.join(PIPELINE_BASE_DIR, SCHEMA_PIPELINE_NAME)

    SCHEMA_METADATA_PATH = os.path.join(
        PIPELINE_BASE_DIR, "metadata", SCHEMA_PIPELINE_NAME, "metadata.db"
    )
    METADATA_PATH = os.path.join(
        PIPELINE_BASE_DIR, "metadata", PIPELINE_NAME, "metadata.db"
    )

    SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, "serving_model", PIPELINE_NAME)

    ENABLE_CACHE = False

    DATA_ROOT = f"{project_root}/data/raw/"
    PREPROCESSING_FN = f"{project_root}/src/pipeline/preprocessing.py"
    TRAINER_MODULE_FILE = f"{project_root}/src/pipeline/train.py"

    # LABELS and FEATURES
    LABEL_KEY = "close_t"
    df_train = pd.read_csv(f"{DATA_ROOT}train_data/train_data.csv")
    FEATURE_KEYS = list(df_train.columns.values)
    FEATURE_KEYS.remove(LABEL_KEY)

    # BATCH_SIZE and STEPS
    TRAIN_BATCH_SIZE = 20
    VAL_BATCH_SIZE = 10
    TEST_BATCH_SIZE = 10
    TRAIN_STEPS = 100
    VAL_STEPS = 5

    # METRICS
    EVAL_ACCURACY_THRESHOLD = 0.6
