# # limitations under the License.
# import os
# import sys

# import tensorflow as tf

# # TODO: code smell of project estructure
# currentdir = os.path.dirname(os.path.realpath(__file__))
# parentdir = os.path.dirname(currentdir)
# parentdir = os.path.dirname(parentdir)
# sys.path.append(parentdir)

# from src.model.model import _make_keras_model


# class ModelTest(tf.test.TestCase):

#   def testBuildKerasModel(self):
#     built_model = _make_keras_model(['foo', 'bar'])
#     self.assertEqual(len(built_model.inputs), 2)

# if __name__ == '__main__':
#   tf.test.main()
