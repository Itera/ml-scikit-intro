import collections
import numpy as np

data = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


def validate_split_and_shuffle_data_set(task):
    output = task.split_and_shuffle_data_set(data, labels)

    assert output is not None, 'a) You need to split the data.'
    assert len(output) == 4, 'a) Return value must be tuple of length four.'

    training_data, test_data, training_labels, test_labels = output

    type_err = 'a) Input type must match output type'
    assert type(data) == type(training_data), type_err
    assert type(data) == type(test_data), type_err
    assert type(labels) == type(training_labels), type_err
    assert type(labels) == type(test_labels), type_err

    assert isinstance(training_data[0], collections.Iterable), 'a) First return value must be subset of data.'
    assert isinstance(test_data[0], collections.Iterable), 'a) Second return value must be subset of data.'
    assert not isinstance(training_labels[0], collections.Iterable), 'a) Third return value must be subset of labels.'
    assert not isinstance(test_labels[0], collections.Iterable), 'a) Fourth return value must be subset of labels.'

    for i in range(0, 10):
        scale = i / 10
        train_d, test_d, train_l, test_l = \
            task.split_and_shuffle_data_set(data, labels, train_proportion=scale)

        assert i == len(train_d), 'a) Length of training data does not match train_proportion'
        assert len(data) - i == len(test_d), 'a) Length of test data does not match train_proportion'
        assert i == len(train_l), 'a) Length of training labels does not match train_proportion'
        assert len(labels) - i == len(test_l), 'a) Length of test labels does not match train_proportion'


def validate_feature_extractor(task):
    extractor = task.FeatureExtractor()

    assert type(extractor.fit(data)) == type(extractor), 'b) Fit method is supposed to return self.'
    assert extractor.transform(data) is not None, 'c) Return the features. See docstring for more details.'


def validate_train_classifier(task):
    clf = task.train_classifier(data, labels)

    assert clf is not None, 'd) Choose estimator (or create your own)'
    assert getattr(clf, "fit", None) is not None, 'd) Estimator must have method fit().'
    assert getattr(clf, "predict", None) is not None, 'd) Estimator must have method predict().'


def approved(task):
    validate_split_and_shuffle_data_set(task)
    validate_feature_extractor(task)
    validate_train_classifier(task)
    return True
