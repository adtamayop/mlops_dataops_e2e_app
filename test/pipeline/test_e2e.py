# import os
# import unittest
# from typing import List

# import ml_metadata as mlmd
# import tensorflow as tf
# from absl import logging
# from absl.testing import parameterized
# from ml_metadata.proto import metadata_store_pb2
# from tfx.examples.penguin import penguin_pipeline_local
# from tfx.v1 import proto
# from tfx.v1.dsl.io import fileio
# from tfx.v1.orchestration import LocalDagRunner, metadata


# @unittest.skipIf(tf.__version__ < '2',
#                  'Uses keras Model only compatible with TF 2.x')
# class PipelineLocalEndToEndTest(tf.test.TestCase,
#                                        parameterized.TestCase):

#   def setUp(self):
#     super().setUp()

#     self._test_dir = os.path.join(
#         os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
#         self._testMethodName)

#     self._pipeline_name = 'penguin_test'
#     self._schema_path = os.path.join(
#         os.path.dirname(__file__), 'schema', 'user_provided', 'schema.pbtxt')
#     self._data_root = os.path.join(os.path.dirname(__file__), 'data')

#     self._serving_model_dir = os.path.join(self._test_dir, 'serving_model')
#     self._pipeline_root = os.path.join(self._test_dir, 'tfx', 'pipelines',
#                                        self._pipeline_name)
#     self._metadata_path = os.path.join(self._test_dir, 'tfx', 'metadata',
#                                        self._pipeline_name, 'metadata.db')




#     _root = os.path.join(".")

#     # _data_root = os.path.join(
#     #     _root,
#     #     Paths.TRAINING_DATA_PATH,
#     #     DownloadDataParams.COMPANY_CODE,
#     #     "train/")

#     _data_root = os.path.join(
#         _root,
#         "data",
#         "test_dataset")

#     _tfx_root = os.path.join(
#         _root,
#         "test",

#      'tfx_pipeline_output')

#     _pipeline_name = 'local_pipeline'
#     _module_file_name = 'model.py'
#     _preprocess_file_name = "preprocessing.py"

#     _module_file = os.path.join(_root, "src", "model", _module_file_name)
#     _preprocessing_file = os.path.join(_root, "src", "preprocessing", _preprocess_file_name)

#     _serving_model_dir = os.path.join(_tfx_root, 'serving_model',
#                                     _pipeline_name)

#     _pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)

#     _metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
#                                 'metadata.db')


#   def _module_file_name(self, model_framework: str) -> str:
#     return os.path.join(
#         os.path.dirname(__file__), f'penguin_utils_{model_framework}.py')


#   def _assertPipelineExecution(self,
#                                has_tuner: bool = False,
#                                has_bulk_inferrer: bool = False,
#                                has_schema_gen: bool = True,
#                                has_pusher: bool = True) -> None:
#     self._assertExecutedOnce('CsvExampleGen')
#     self._assertExecutedOnce('Evaluator')
#     self._assertExecutedOnce('ExampleValidator')
#     self._assertExecutedOnce('StatisticsGen')
#     self._assertExecutedOnce('Trainer')
#     self._assertExecutedOnce('Transform')
#     if has_schema_gen:
#       self._assertExecutedOnce('SchemaGen')
#     else:
#       self._assertExecutedOnce('ImportSchemaGen')
#       self._assertExecutedOnce('Pusher')


#   @parameterized.parameters(
#       ('keras'))


#   def testPenguinPipelineLocal(self, model_framework):
#     module_file = self._module_file_name(model_framework)
#     pipeline = penguin_pipeline_local._create_pipeline(
#         pipeline_name=self._pipeline_name,
#         data_root=self._data_root,
#         module_file=module_file,
#         accuracy_threshold=0.1,
#         serving_model_dir=self._serving_model_dir,
#         pipeline_root=self._pipeline_root,
#         metadata_path=self._metadata_path,
#         user_provided_schema_path=None,
#         enable_tuning=False,
#         enable_bulk_inferrer=False,
#         examplegen_input_config=None,
#         examplegen_range_config=None,
#         resolver_range_config=None,
#         enable_transform_input_cache=False)

#     logging.info('Starting the first pipeline run.')
#     LocalDagRunner().run(pipeline)

#     self.assertTrue(fileio.exists(self._serving_model_dir))
#     self.assertTrue(fileio.exists(self._metadata_path))
#     expected_execution_count = 9  # 8 components + 1 resolver
#     metadata_config = metadata.sqlite_metadata_connection_config(
#         self._metadata_path)
#     store = mlmd.MetadataStore(metadata_config)
#     artifact_count = len(store.get_artifacts())
#     execution_count = len(store.get_executions())
#     self.assertGreaterEqual(artifact_count, execution_count)
#     self.assertEqual(expected_execution_count, execution_count)

#     self._assertPipelineExecution()

#     logging.info('Starting the second pipeline run. All components except '
#                  'Evaluator and Pusher will use cached results.')
#     LocalDagRunner().run(pipeline)

#     # Artifact count is increased by 3 caused by Evaluator and Pusher.
#     self.assertLen(store.get_artifacts(), artifact_count + 3)
#     artifact_count = len(store.get_artifacts())
#     self.assertLen(store.get_executions(), expected_execution_count * 2)

#     logging.info('Starting the third pipeline run. '
#                  'All components will use cached results.')
#     LocalDagRunner().run(pipeline)

#     # Asserts cache execution.
#     # Artifact count is unchanged.
#     self.assertLen(store.get_artifacts(), artifact_count)
#     self.assertLen(store.get_executions(), expected_execution_count * 3)



# if __name__ == '__main__':
#   tf.compat.v1.enable_v2_behavior()
#   tf.test.main()
