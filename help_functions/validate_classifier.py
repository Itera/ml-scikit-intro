import numpy as np
from sklearn import metrics
from help_functions import result_printer


def validate_model(clf, test_src: iter, test_data: iter, test_labels: iter, top: int = 5, average: str = 'micro',
                   pos_label: int = 1, bitmap: bool = False):
    predictions = clf.predict(test_data)

    classification_report = metrics.classification_report(test_labels, predictions)
    confusion_matrix = metrics.confusion_matrix(test_labels, predictions)

    multi_class = len(np.unique(test_labels, return_counts=True)[0]) > 2

    if multi_class:
        f1_score = metrics.f1_score(test_labels, predictions, average=average)
        precision = metrics.precision_score(test_labels, predictions, average=average)
        recall = metrics.recall_score(test_labels, predictions, average=average)
    else:
        f1_score = metrics.f1_score(test_labels, predictions, pos_label=pos_label)
        precision = metrics.precision_score(test_labels, predictions, pos_label=pos_label)
        recall = metrics.recall_score(test_labels, predictions, pos_label=pos_label)

    print('Classification report:\n%s\n\nConfusion matrix:\n%s\n\nF-score: %.2f\nPrecision: %.2f\nRecall: %.2f\n'
          % (classification_report, confusion_matrix, f1_score, precision, recall))

    result_printer.print_wrong_predictions(test_src, predictions, test_labels, top, bitmap=bitmap)
