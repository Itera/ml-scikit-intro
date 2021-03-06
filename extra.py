from help_functions import validate_tasks, data_retriever
from tasks import number_classifier

"""
Parameters
----------
show: int
    Decides number of wrongly predicted labels to show (all=-1)
rows: int
    Decides number of data samples to use (all=-1)
"""

if __name__ == '__main__':
    if validate_tasks.approved(number_classifier, data_retriever.load_mnist):
        number_classifier.run_number_classifier()
