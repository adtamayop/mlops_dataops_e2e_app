# import os
# import sys

# import tensorflow as tf

# # TODO: code smell of project estructure
# currentdir = os.path.dirname(os.path.realpath(__file__))
# parentdir = os.path.dirname(currentdir)
# # parentdir = os.path.dirname(parentdir)
# sys.path.append(parentdir)
# # from src.pipeline import preprocessing  # NOQA: E402 (tengo que importarlo así)


# class UtilsTest(tf.test.TestCase):
#     def setUp(self):
#         super(UtilsTest, self).setUp()
#         # self._testdata_path = os.path.join(
#         # TEST_COMPONENTS_PATH, 'components/testdata')
#         key = "fare"
#         xfm_key = preprocessing.transformed_name(key)
#         self.assertEqual(xfm_key, "fare_xf")

#     def testUtils(self):
#         pass


# if __name__ == "__name__":
#     # UtilsTest()
#     pass
