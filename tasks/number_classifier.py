import numpy as np
from sklearn.base import TransformerMixin
from help_functions import data_retriever
from help_functions.validate_classifier import validate_model


class BitmapFeatureExtractor(TransformerMixin):
    """
    Extracts features from the bitmap.
    """

    def fit(self, bitmaps: iter, *others):
        """
        Use the data to prepare for any transformation.
        :param bitmaps: A list of bitmaps.
        :param others: Stuff other modules might need.
        :return: The Transformer itself. This allows for method-chaining.
        """
        return
        return self

    def transform(self, bitmaps, *others):
        """
        Transform the bitmaps to wanted representation.
        :param bitmaps: A list of bitmaps.
        :param others: Stuff other modules might need.
        :return: The extracted features.
        """
        return


def split_and_shuffle_data_set(data: np.ndarray, labels: np.ndarray, train_proportion: float = 0.8):
    return


def train_classifier(training_features, training_labels):
    return


def run_number_classifier():
    rows = -1  # -1 means retrieving complete set. When testing, set lower for faster training (e.g. 5000).
    print('-- Executing number classification')

    print('Loading data...')
    data, labels = data_retriever.load_mnist(rows)

    print('Splitting data...')
    training_data, test_data, training_labels, test_labels = split_and_shuffle_data_set(data, labels)

    print('Extracting features...')
    extractor = BitmapFeatureExtractor()
    extractor.fit(training_data)
    training_features = extractor.transform(training_data)
    test_features = extractor.transform(test_data)

    print('Training classifier...')
    classifier = train_classifier(training_features, training_labels)

    print('Testing classifier...')
    validate_model(classifier, test_data, test_features, test_labels, bitmap=True)
