import csv
import numpy as np
import os
import pickle
from sklearn import datasets, utils

REVIEW_SOURCE = './files/review_data/review_source.csv'
REVIEW_DATA = './files/review_data/review_data_%d.sav'
SMS_SOURCE = './files/spam_data/sms_source.csv'
SMS_DATA = './files/spam_data/sms_data_%d.sav'


def __read_file(source, rows):
    data_set = []
    with open(source, 'r', encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        csv_reader.__next__()

        for i, row in enumerate(csv_reader):
            if i >= rows >= 0:
                break

            data_set.append(row)

    return np.array(data_set)


def __load_file(source_location: str, save_location: str, cache_data: bool, rows: int):
    print('Retrieving data...')

    if os.path.exists(save_location) and cache_data:
        return pickle.load(open(save_location, 'rb'))

    data_set = __read_file(source_location, rows)

    if cache_data:
        pickle.dump(data_set, open(save_location, 'wb'))

    return data_set


def load_reviews(cache_data: bool = False, rows: int = -1):
    data_set = __load_file(REVIEW_SOURCE, REVIEW_DATA % rows, cache_data, rows)
    return data_set[:, -1], data_set[:, 0].astype(int), data_set[:, 1].astype(int)


def load_sms(cache_data: bool = False, rows: int = -1):
    data_set = __load_file(SMS_SOURCE, SMS_DATA % rows, cache_data, rows)
    return data_set[:, -1], data_set[:, 0].astype(int)


def load_mnist(rows: int = -1):
    mnist = datasets.fetch_mldata('MNIST original', data_home='./files')
    print("Fetched %d bitmaps." % len(mnist.target))

    print("Shuffle data set")
    mnist.data, mnist.target = utils.shuffle(mnist.data, mnist.target)

    if rows < 0:
        return mnist.data, mnist.target
    else:
        return mnist.data[:rows], mnist.target[:rows]
