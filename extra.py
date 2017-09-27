from tasks.number_classifier import run_number_classifier
from unit_tests import test_number_classifier

"""
Parameters
----------
show: int
    Decides number of wrongly predicted labels to show (all=-1)
rows: int
    Decides number of data samples to use (all=-1)
"""

if __name__ == '__main__':
    if test_number_classifier.approved():
        run_number_classifier()
