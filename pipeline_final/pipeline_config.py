import os
import sys

# TODO: Modify project structure for don't do this smell code
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


from src.config.config import DownloadDataParams, Paths

# execution_type = "CLOUD"
execution_type = "LOCAL"

if execution_type == "LOCAL":

    _root = os.path.join(".")

    # _data_root = os.path.join(
    #     _root,
    #     Paths.TRAINING_DATA_PATH,
    #     DownloadDataParams.COMPANY_CODE,
    #     "train/")

    _data_root = os.path.join(
        _root,
        Paths.TRAINING_DATA_PATH,
        DownloadDataParams.COMPANY_CODE)

    _tfx_root = os.path.join(_root, 'tfx_pipeline_output')

    _examplegen_input_config = None
    _examplegen_range_config = None
    _resolver_range_config = None

    _pipeline_name = 'local_pipeline'
    _module_file_name = 'model.py'
    _preprocess_file_name = "preprocessing.py"

    _module_file = os.path.join(_root, "src", "model", _module_file_name)
    _preprocessing_file = os.path.join(_root, "src", "preprocessing", _preprocess_file_name)

    _serving_model_dir = os.path.join(_tfx_root, 'serving_model',
                                    _pipeline_name)

    _pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)

    _metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                                'metadata.db')

elif execution_type == "CLOUD":
    pass
