import os
import shutil
import sys
from typing import Text

import tensorflow as tf
import tfx
from absl.testing import parameterized
from tfx import v1 as tfx
from tfx.dsl.io import fileio
from tfx.orchestration import metadata
from tfx.orchestration.local.local_dag_runner import LocalDagRunner

# TODO: code smell of project estructure
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)

parentdir_ = os.path.dirname(parentdir)
sys.path.append(parentdir_)


from pipeline_final.pipeline import _create_pipeline
from pipeline_final.pipeline_config import (_data_root, _metadata_path,
                                            _module_file, _pipeline_name,
                                            _pipeline_root,
                                            _preprocessing_file, _root,
                                            _serving_model_dir)
from src.config.config import Features, Paths
from src.preprocessing.preprocessing import transformed_name

# import pipeline_local  # NOQA: E402 (tengo que importarlo así)

# from src.config.pipeling_config import (  # NOQA: E402 (tengo que importarlo así)
#     DATA_ROOT, PIPELINE_NAME, PREPROCESSING_FN, SERVING_MODEL_DIR,
#     TEST_PIPELINE_PATH, TRAINER_MODULE_FILE)
TEST_PIPELINE_PATH = f"test/pipeline/testPipeline/"


class PipelineEndToEndTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        super(PipelineEndToEndTest, self).setUp()
        self._test_dir = os.path.join(TEST_PIPELINE_PATH)
        self._pipeline_name = _pipeline_name
        self._data_root = Paths.TEST_DATA_PATH
        self._module_file = _module_file
        self._preprocessing_file = _preprocessing_file
        self._serving_model_dir = _serving_model_dir
        self._pipeline_root = os.path.join(self._test_dir,"pipelines", self._pipeline_name)
        self._metadata_path = os.path.join(
            self._test_dir,"metadata", self._pipeline_name, "metadata.db"
        )

        self.clean_directory(os.path.join(self._test_dir,"pipelines"))
        self.clean_directory(os.path.join(self._test_dir,"metadata"))

    def clean_directory(self, path_to_clean):
        list_dir = os.listdir(path_to_clean)
        for filename in list_dir:
            if filename == ".gitignore":
                continue
            else:
                file_path = os.path.join(path_to_clean, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    print("deleting file:", file_path)
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    print("deleting folder:", file_path)
                    shutil.rmtree(file_path)

    def assertExecutedOnce(self, component: Text) -> None:
        """Check the component is executed exactly once."""
        component_path = os.path.join(self._pipeline_root, component)
        self.assertTrue(fileio.exists(component_path))
        outputs = fileio.listdir(component_path)

        self.assertIn(".system", outputs)
        outputs.remove(".system")
        system_paths = [
            os.path.join(".system", path)
            for path in fileio.listdir(os.path.join(component_path, ".system"))
        ]
        self.assertNotEmpty(system_paths)
        self.assertIn(".system/executor_execution", system_paths)
        outputs.extend(system_paths)
        self.assertNotEmpty(outputs)
        for output in outputs:
            execution = fileio.listdir(os.path.join(component_path, output))
        self.assertLen(execution, 1)

    def assertPipelineExecution(self) -> None:
        self.assertExecutedOnce("CsvExampleGen")
        self.assertExecutedOnce("Evaluator")
        self.assertExecutedOnce("ExampleValidator")
        self.assertExecutedOnce("Pusher")
        self.assertExecutedOnce("SchemaGen")
        self.assertExecutedOnce("StatisticsGen")
        self.assertExecutedOnce("Trainer")
        self.assertExecutedOnce("Transform")


    def testPipeline(self):
        tfx.orchestration.LocalDagRunner().run(
            _create_pipeline(
                pipeline_name=_pipeline_name,
                pipeline_root=self._pipeline_root,
                data_root=_data_root,
                module_file=_module_file,
                preprocess_file = _preprocessing_file,
                accuracy_threshold=0.1,
                serving_model_dir=self._serving_model_dir,
                metadata_path=self._metadata_path,
                user_provided_schema_path=None,
                enable_tuning=True,
                enable_bulk_inferrer=True,
                enable_transform_input_cache=True))

        self.assertTrue(fileio.exists(self._serving_model_dir))
        self.assertTrue(fileio.exists(self._metadata_path))
        metadata_config = metadata.sqlite_metadata_connection_config(self._metadata_path)

        with metadata.Metadata(metadata_config) as m:
            artifact_count = len(m.store.get_artifacts())
            execution_count = len(m.store.get_executions())
            self.assertGreaterEqual(artifact_count, execution_count)
            self.assertEqual(10, execution_count)

        self.assertPipelineExecution()


if __name__ == "__main__":
    tf.test.main()
