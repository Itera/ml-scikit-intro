from sklearn.base import ClassifierMixin, TransformerMixin
import numpy as np
from help_functions import data_retriever
from help_functions.validate_classifier import validate_model


class SMSFeatureExtractor(TransformerMixin):
    """
    We are going to make a Transformer.
    Transformers are generally used for two things:
        1. Extract features from raw data (such as a list of SMSes).
        2. Transform a feature set into another feature set.
    """

    def fit(self, documents: iter, *others):
        """
        The goal of this method is to do review the data and prepare for any transformation.
        For this task we are going to make a word bag model. Because the fit method prepares a transformer, so that it
        can transform any given data, you should store the available words.
        :param documents: A list of text messages.
        :param others: Stuff other scikit-learn modules might tack on, that we will ignore.
        :return: The Transformer itself. This allows for method-chaining.
        """
        raise NotImplementedError('You should store all the distinct words here.')
        return self

    def transform(self, documents, *others):
        """
        This method is where we do the feature extraction. It is called transform because we are
        transforming the data from one representation to another. See the readme for an input/output table.
        :param documents:  A list of text messages.
        :param others: Stuff other scikit-learn modules might tack on, that we will ignore.
        :return: An NxM matrix where N is the amount of text messages and M is the amount of features (words).
        """
        raise NotImplementedError('Return the features. See docstring for more details.')


class ExpectedValueClassifier(ClassifierMixin):
    """
    The stupidest estimator I could think of.
    Finds the most commonly occurring label and returns that for all the samples.
    """

    def __init__(self):
        self.most_frequent_class = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This is where we train the model on the data.
        Here we are just going to count the number of occurrences of each label, storing the most frequent.
        :param X: The features. We ignore those here.
        :param y: The labels, which we will be counting.
        :return:
        """
        minimum = np.min(y)
        offset = 0
        if minimum < 0:
            offset = np.abs(minimum)
        histogram = np.bincount(y + offset)
        self.most_frequent_class = histogram.argmax() - offset

        return self

    def predict(self, X: np.ndarray):
        """
        This method predicts labels for the given features based on its training.
        :param X: A matrix of size (m,n). The features of the data we want to predict.
        :return: A vector of labels. Length m.
        """
        return np.asarray([self.most_frequent_class] * X.shape[0])


def split_and_shuffle_data_set(data: np.ndarray, labels: np.ndarray, train_proportion: float=0.8):
    raise NotImplementedError("You need to split the data! Look at the powerpoint or raise your hand if you are stuck :)")


def train_classifier(training_features, training_labels):
    return ExpectedValueClassifier().fit(training_features, training_labels)


def run_spam_filter():
    row_count = -1  # set this number to some number below 2000 if you are having performance problems
    print('-- Executing spam filter')
    print('-- Loading data')
    data, labels = data_retriever.load_sms(cache_data=False, rows=row_count)

    # randomize and split the data
    training_data, training_labels, test_data, test_labels = split_and_shuffle_data_set(data, labels)

    # fit the transformer
    extractor = SMSFeatureExtractor()
    extractor.fit(training_data)

    # extract the features from the test data
    training_features = extractor.transform(training_data)

    # train the classifier
    classifier = train_classifier(training_features, training_labels)

    # generate classification report
    test_features = extractor.transform(test_data)
    validate_model(classifier, test_data, test_features, test_labels)
