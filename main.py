from help_functions import validate_regression
from tasks.spam_filter import run_spam_filter
"""
Parameters
----------
cache_data: bool
    Decides if data is to be cached (full review data may uses a lot of space, this may cause som problems on Mac)
rows: int
    Decides number of data samples to use (all=-1)
"""

if __name__ == '__main__':
    run_spam_filter()
    print("If you are satisfied with the results, uncomment the guy underneath me to start task 2. (im in main.py :D)")
    # validate_regression.execute_sentiment_analysis(cache_data=False, rows=5000)
