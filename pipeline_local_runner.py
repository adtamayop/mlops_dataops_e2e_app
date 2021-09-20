import os
import sys

import absl
from tfx import v1 as tfx

from pipeline_final.pipeline import _create_pipeline
from pipeline_final.pipeline_config import (_data_root, _metadata_path,
                                            _module_file, _pipeline_name,
                                            _pipeline_root,
                                            _preprocessing_file,
                                            _serving_model_dir)
from src.config.config import Features
from src.preprocessing.preprocessing import transformed_name

if __name__ == '__main__':

    absl.logging.set_verbosity(absl.logging.INFO)
    absl.flags.FLAGS(sys.argv)

    tfx.orchestration.LocalDagRunner().run(
        _create_pipeline(
            pipeline_name=_pipeline_name,
            pipeline_root=_pipeline_root,
            data_root=_data_root,
            module_file=_module_file,
            preprocess_file = _preprocessing_file,
            accuracy_threshold=0.1,
            serving_model_dir=_serving_model_dir,
            metadata_path=_metadata_path,
            user_provided_schema_path=None,
            enable_tuning=True,
            enable_bulk_inferrer=True,
            enable_transform_input_cache=True))
