import collections
import numpy as np

data = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


def validate_split_and_shuffle_data_set(task):
    output = task.split_and_shuffle_data_set(data, labels)
    name = 'split_and_shuffle_data_set'

    assert output is not None, '%s: You need to split the data.' % name
    assert len(output) == 4, '%s: Return value must be tuple of length four.' % name

    training_data, test_data, training_labels, test_labels = output

    type_err = '%s: Input type must match output type' % name
    assert type(data) == type(training_data), type_err
    assert type(data) == type(test_data), type_err
    assert type(labels) == type(training_labels), type_err
    assert type(labels) == type(test_labels), type_err

    assert isinstance(training_data[0], collections.Iterable), '%s: First return value must be subset of data.' % name
    assert isinstance(test_data[0], collections.Iterable), '%s: Second return value must be subset of data.' % name
    assert not isinstance(training_labels[0], collections.Iterable), '%s: Third return value must be subset of labels.' % name
    assert not isinstance(test_labels[0], collections.Iterable), '%s: Fourth return value must be subset of labels.' % name

    for i in range(0, 10):
        scale = i / 10
        train_d, test_d, train_l, test_l = \
            task.split_and_shuffle_data_set(data, labels, train_proportion=scale)

        assert i == len(train_d), '%s: Length of training data does not match train_proportion' % name
        assert len(data) - i == len(test_d), '%s: Length of test data does not match train_proportion' % name
        assert i == len(train_l), '%s: Length of training labels does not match train_proportion' % name
        assert len(labels) - i == len(test_l), '%s: Length of test labels does not match train_proportion' % name


def validate_feature_extractor(task):
    extractor = task.FeatureExtractor()
    name = 'FeatureExtractor'

    assert type(extractor.fit(data)) == type(extractor), '%s: Fit method is supposed to return self.' % name
    assert extractor.transform(data) is not None, '%s: Return the features. See docstring for more details.' % name


def validate_train_classifier(task):
    clf = task.train_classifier(data, labels)
    name = 'train_classifier'

    assert clf is not None, '%s: Choose estimator (or create your own)' % name
    assert getattr(clf, "fit", None) is not None, '%s: Estimator must have method fit().' % name
    assert getattr(clf, "predict", None) is not None, '%s: Estimator must have method predict().' % name


def approved(task):
    validate_split_and_shuffle_data_set(task)
    validate_feature_extractor(task)
    validate_train_classifier(task)
    return True
