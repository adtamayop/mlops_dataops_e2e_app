# import os
# import shutil
# import sys
# from typing import Text

# import tensorflow as tf
# from absl.testing import parameterized
# from tfx.dsl.io import fileio
# from tfx.orchestration import metadata
# from tfx.orchestration.local.local_dag_runner import LocalDagRunner

# # TODO: code smell of project estructure
# currentdir = os.path.dirname(os.path.realpath(__file__))
# parentdir = os.path.dirname(currentdir)
# # parentdir = os.path.dirname(parentdir)
# sys.path.append(parentdir)

# import pipeline_local  # NOQA: E402 (tengo que importarlo así)
# from src.config.pipeling_config import (  # NOQA: E402 (tengo que importarlo así)
#     DATA_ROOT,
#     PIPELINE_NAME,
#     PREPROCESSING_FN,
#     SERVING_MODEL_DIR,
#     TEST_PIPELINE_PATH,
#     TRAINER_MODULE_FILE,
# )


# class PipelineEndToEndTest(tf.test.TestCase, parameterized.TestCase):
#     def setUp(self):
#         super(PipelineEndToEndTest, self).setUp()
#         self._test_dir = os.path.join(TEST_PIPELINE_PATH, self._testMethodName)
#         self._pipeline_name = PIPELINE_NAME
#         self._data_root = DATA_ROOT
#         self._module_file = TRAINER_MODULE_FILE
#         self._preprocessing_file = PREPROCESSING_FN
#         self._serving_model_dir = SERVING_MODEL_DIR
#         self._pipeline_root = os.path.join(self._test_dir,"pipelines", self._pipeline_name)
#         self._metadata_path = os.path.join(
#             self._test_dir,"metadata", self._pipeline_name, "metadata.db"
#         )

#         self.clean_directory(os.path.join(self._test_dir,"pipelines"))
#         self.clean_directory(os.path.join(self._test_dir,"metadata"))

#     def clean_directory(self, path_to_clean):
#         list_dir = os.listdir(path_to_clean)
#         for filename in list_dir:
#             if filename == ".gitignore":
#                 continue
#             else:
#                 file_path = os.path.join(path_to_clean, filename)
#                 if os.path.isfile(file_path) or os.path.islink(file_path):
#                     print("deleting file:", file_path)
#                     os.unlink(file_path)
#                 elif os.path.isdir(file_path):
#                     print("deleting folder:", file_path)
#                     shutil.rmtree(file_path)



#     def assertExecutedOnce(self, component: Text) -> None:
#         """Check the component is executed exactly once."""
#         component_path = os.path.join(self._pipeline_root, component)
#         self.assertTrue(fileio.exists(component_path))
#         outputs = fileio.listdir(component_path)

#         self.assertIn(".system", outputs)
#         outputs.remove(".system")
#         system_paths = [
#             os.path.join(".system", path)
#             for path in fileio.listdir(os.path.join(component_path, ".system"))
#         ]
#         self.assertNotEmpty(system_paths)
#         self.assertIn(".system/executor_execution", system_paths)
#         outputs.extend(system_paths)
#         self.assertNotEmpty(outputs)
#         for output in outputs:
#             execution = fileio.listdir(os.path.join(component_path, output))
#         self.assertLen(execution, 1)

#     def assertPipelineExecution(self) -> None:
#         self.assertExecutedOnce("CsvExampleGen")
#         self.assertExecutedOnce("Evaluator")
#         self.assertExecutedOnce("ExampleValidator")
#         self.assertExecutedOnce("Pusher")
#         self.assertExecutedOnce("SchemaGen")
#         self.assertExecutedOnce("StatisticsGen")
#         self.assertExecutedOnce("Trainer")
#         self.assertExecutedOnce("Transform")
#         self.assertExecutedOnce("CsvExampleGen.bulk_example")
#         self.assertExecutedOnce("BulkInferrer.bulk_infer")
#         # self.assertExecutedOnce('serving_model')

#     def testPipeline(self):
#         LocalDagRunner().run(
#             pipeline_local._create_pipeline(
#                 pipeline_name=self._pipeline_name,
#                 data_root=self._data_root,
#                 module_file=self._module_file,
#                 preprocessing_file=self._preprocessing_file,
#                 serving_model_dir=self._serving_model_dir,
#                 pipeline_root=self._pipeline_root,
#                 metadata_path=self._metadata_path,
#             )
#         )

#         self.assertTrue(fileio.exists(self._serving_model_dir))
#         self.assertTrue(fileio.exists(self._metadata_path))
#         metadata_config = metadata.sqlite_metadata_connection_config(self._metadata_path)

#         with metadata.Metadata(metadata_config) as m:
#             artifact_count = len(m.store.get_artifacts())
#             execution_count = len(m.store.get_executions())
#             self.assertGreaterEqual(artifact_count, execution_count)
#             self.assertEqual(11, execution_count)

#         self.assertPipelineExecution()

#         # # Runs pipeline the second time.
#         # LocalDagRunner().run(
#         #     pipeline_local._create_pipeline(
#         #         pipeline_name=self._pipeline_name,
#         #         data_root=self._data_root,
#         #         module_file=self._module_file,
#         #         preprocessing_file=self._preprocessing_file,
#         #         serving_model_dir=self._serving_model_dir,
#         #         pipeline_root=self._pipeline_root,
#         #         metadata_path=self._metadata_path,
#         #     )
#         # )

#         # # All executions but Evaluator and Pusher are cached.
#         # # Note that Resolver will always execute.

#         # with metadata.Metadata(metadata_config) as m:
#         #     # Artifact count is increased by 3 caused by Evaluator and Pusher.
#         #     self.assertLen(m.store.get_artifacts(), artifact_count + 4)
#         #     artifact_count = len(m.store.get_artifacts())
#         #     self.assertLen(m.store.get_executions(), 22)


# if __name__ == "__main__":
#     tf.test.main()
