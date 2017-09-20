from help_functions import data_retriever
from tasks import sentiment_analysis
from sklearn.model_selection import train_test_split
import numpy as np
from random import sample


def __rmse(y_predicted, y_target):
    return np.sqrt(((y_predicted - y_target) ** 2).mean())


def __validate_model(pipeline, training_data: iter, test_data: iter, training_labels: iter, test_labels: iter):
    pipeline.fit(training_data, training_labels)

    predicted_labels = pipeline.predict(test_data)

    score = __rmse(predicted_labels, test_labels)
    random_labels = np.random.randint(low=1, high=11, size=len(predicted_labels))
    random_score = __rmse(random_labels, test_labels)

    print('Done. Your RMSE was : {}'.format(score))
    print('For reference just guessing would have yielded: {}'.format(random_score))

    if score < 2:
        print('Wow, well done!')
    elif score < 2.25:
        print('You are officially a wizard. You can keep banging your head, or move on to the images :)')
    elif score < 2.7:
        print('Getting below 2.7 is not easy. Good job! You can keep banging your head, or move on to the images :)')
    elif score < 3:
        print('Keep going. You are below three. Try tuning some more!')
    elif score < 4:
        print('Moving in the right direction! You are beating a stupid guesser.')
    else:
        print('Ok! It works :D Remember that you can always ask for help, or read the docs, if you get stuck.')

    return pipeline, predicted_labels


def __display_results(data, target, predicted):
    miss_size = list(zip(data, target, predicted, list(np.abs(target - predicted))))

    example_count = 5

    print('-------- LETS LOOK AT SOME EXAMPLES --------')
    for example in sample(miss_size, example_count):
        print('Target: {1} Predicted: {2} Miss: {3} \nText: {0}'.format(*example))


def execute_sentiment_analysis(rows: int = -1, cache_data: bool = False):
    print('-- Executing sentiment analysis')
    data, labels, ratings = data_retriever.load_reviews(rows=rows, cache_data=cache_data)

    pipeline = sentiment_analysis.make_pipeline()

    print('Splitting data...')
    training_data, test_data, training_labels, test_labels = train_test_split(data, ratings)
    print('- size: %d (training), %d (test)' % (len(training_labels), len(test_labels)))

    pipeline, predictions = __validate_model(pipeline, training_data, test_data, training_labels, test_labels)

    __display_results(test_data, test_labels, predictions)
