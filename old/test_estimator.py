# """
# Testing file for a combined KMeans/PLSR estimator.
# """
# # import unittest
# from sklearn.datasets import load_diabetes
# from ..estimator import kmeansPLSR
# import random
# import string


# class TestEstimator(unittest.TestCase):
#     """ Testing class for a combined KMeans/PLSR estimator. """

#     def test_fitting(self):
#         """ Check that we can in fact perform fitting in one case. """
#         X, y = load_diabetes(return_X_y=True)

#         MadeUp_ProtNames, MadeUp_peptide_phosphosite = [], []
#         for i in range(X.shape[0]):
#             MadeUp_ProtNames.append(''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10)))
#             MadeUp_peptide_phosphosite.append(''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10)))

# #         frame = [pd.Series(MadeUp_peptide_phosphosite), pd.Series(MadeUp_ProtNames), pd.DataFrame(X)]
# #         X = pd.concat(frame, axis = 1)

#         est = kmeansPLSR(3, 2, MadeUp_ProtNames, MadeUp_peptide_phosphosite)

#         est.fit(X, y)
