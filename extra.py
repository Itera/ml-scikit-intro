from tasks import number_classifier
from unit_tests import validate_tasks

"""
Parameters
----------
show: int
    Decides number of wrongly predicted labels to show (all=-1)
rows: int
    Decides number of data samples to use (all=-1)
"""

if __name__ == '__main__':
    if validate_tasks.approved(number_classifier):
        number_classifier.run_number_classifier()
