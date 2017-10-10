from help_functions import validate_regression
from tasks import spam_filter
from unit_tests import validate_tasks

"""
Parameters
----------
cache_data: bool
    Decides if data is to be cached (full review data may uses a lot of space, this may cause som problems on Mac)
rows: int
    Decides number of data samples to use (all=-1)
"""

if __name__ == '__main__':
    if validate_tasks.approved(spam_filter):
        spam_filter.run_spam_filter()
        print("If you are satisfied with the results, uncomment next line to start task 2. (in main.py)")
        # validate_regression.execute_sentiment_analysis(cache_data=False, rows=5000)
