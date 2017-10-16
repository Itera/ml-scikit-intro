import collections
import numpy as np

source = {0: [0, 0], 1: [1, 1], 2: [2, 2], 3: [3, 3], 4: [4, 4], 5: [5, 5], 6: [6, 6], 7: [7, 7], 8: [8, 8], 9: [9, 9]}
data = np.array(list(source.values()))
labels = np.array(list(source.keys()))


def __validate_binding(feature, label, name):
    for i, lbl in enumerate(label):
        assert source[lbl][0] == feature[i][0], '%s: Binding between feature and label must be preserved.' % name
        assert source[lbl][1] == feature[i][1], '%s: Binding between feature and label must be preserved.' % name


def __validate_split_and_shuffle_data_set(task):
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

    assert len(training_data) == len(training_labels), '%s: Data and labels must have same length.' % name
    assert len(test_data) == len(test_labels), '%s: Data and labels must have same length.' % name

    __validate_binding(training_data, training_labels, name)
    __validate_binding(test_data, test_labels, name)

    for i in range(0, 10):
        scale = i / 10
        train_d, test_d, train_l, test_l = \
            task.split_and_shuffle_data_set(data, labels, train_proportion=scale)

        assert i == len(train_d), '%s: Length of training data does not match train_proportion' % name
        assert len(data) - i == len(test_d), '%s: Length of test data does not match train_proportion' % name
        assert i == len(train_l), '%s: Length of training labels does not match train_proportion' % name
        assert len(labels) - i == len(test_l), '%s: Length of test labels does not match train_proportion' % name


def __validate_feature_extractor(task, real_data):
    extractor = task.FeatureExtractor()
    name = 'FeatureExtractor'

    assert type(extractor.fit(real_data)) == type(extractor), '%s: Fit method is supposed to return self.' % name
    assert extractor.transform(real_data) is not None, '%s: Return the features. See docstring for more details.' % name


def __validate_train_classifier(task):
    clf = task.train_classifier(data, labels)
    name = 'train_classifier'

    assert clf is not None, '%s: Choose estimator (or create your own)' % name
    assert getattr(clf, "fit", None) is not None, '%s: Estimator must have method fit().' % name
    assert getattr(clf, "predict", None) is not None, '%s: Estimator must have method predict().' % name

    clf.predict(data)


def approved(task, data_retriever):
    real_data = data_retriever(rows=10)
    __validate_split_and_shuffle_data_set(task)
    __validate_feature_extractor(task, real_data)
    __validate_train_classifier(task)
    return True
