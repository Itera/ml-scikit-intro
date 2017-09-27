import unittest
import numpy as np
from sklearn.exceptions import NotFittedError
from tasks import number_classifier

data = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


class NumberClassifierTest(unittest.TestCase):
    def test_split_and_shuffle_data_set(self):
        output = number_classifier.split_and_shuffle_data_set(data, labels)

        self.assertIsNotNone(output, msg='a) You need to split the data.')
        self.assertEqual(len(output), 4, msg='a) Return value must be tuple of length four.')

        training_data, test_data, training_labels, test_labels = output

        type_err = 'a) Input type must match output type'
        self.assertEqual(type(data), type(training_data), msg=type_err)
        self.assertEqual(type(data), type(test_data), msg=type_err)
        self.assertEqual(type(labels), type(training_labels), msg=type_err)
        self.assertEqual(type(labels), type(test_labels), msg=type_err)

        order_err = 'a) Return value must be in correct order (training_data, test_data, training_labels, test_labels)'
        self.assertEqual(len(data[0]), len(training_data[0]), msg=order_err)
        self.assertEqual(len(data[0]), len(test_data[0]), msg=order_err)

        scale_err = 'a) Length of return values does not match train_proportion'
        for i in range(0, 10):
            scale = i / 10
            train_d, test_d, train_l, test_l = \
                number_classifier.split_and_shuffle_data_set(data, labels, train_proportion=scale)

            self.assertEqual(i, len(train_d), msg=scale_err)
            self.assertEqual(i, len(train_l), msg=scale_err)
            self.assertEqual(len(data)-i, len(test_d), msg=scale_err)
            self.assertEqual(len(labels)-i, len(test_l), msg=scale_err)

    def test_feature_extractor(self):
        extractor = number_classifier.BitmapFeatureExtractor()

        implement_err_b = 'b) You may store som relevant information about the data set here.'
        self.assertIsNotNone(extractor.fit(data), msg=implement_err_b)

        implement_err_c = 'c) Return the features. See docstring for more details.'
        self.assertIsNotNone(extractor.transform(data), msg=implement_err_c)

    def test_train_classifier(self):
        clf = number_classifier.train_classifier(data, labels)

        self.assertIsNotNone(clf, msg='d) Choose estimator (or create your own), fit it and return it.')
        self.assertIsNotNone(getattr(clf, "fit", None), msg='d) Estimator must have method fit().')
        self.assertIsNotNone(getattr(clf, "predict", None), msg='d) Estimator must have method predict().')

        try:
            clf.predict(data)
        except NotFittedError:
            self.fail("d) Estimator is not fitted/trained.")


def approved():
    runner = unittest.TextTestRunner()
    result = runner.run(unittest.makeSuite(NumberClassifierTest))

    return len(result.errors) == 0 and len(result.failures) == 0
