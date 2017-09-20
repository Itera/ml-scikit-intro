from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.base import TransformerMixin, RegressorMixin


def make_pipeline():
    """
    This is where we will build our pipeline.
    Pipelines are basically a chain of transformers followed by an estimator.
    The first transformer should be one (or more) methods of feature extraction.

    :return: a working pipeline.
    """
    pipeline_steps = [
        ('sfe', StupidFeatureExtractor()),
        ('svr', RandomRegressor())
    ]

    return Pipeline(steps=pipeline_steps)


class StupidFeatureExtractor(TransformerMixin):
    """
    Just a stupid transformer that takes in some data and returns garbage.
    """

    def __init__(self):
        """
        This is where you would accept the hyper parameters
        """
        pass

    def fit(self, documents, y=None):
        return self

    def transform(self, documents, y=None):
        """
        This does not actually transform the documents to anything.
        It just spits out a random matrix with the correct amount of rows and 5 cols.
        """
        return np.random.rand(len(documents), 5)


class RandomRegressor(RegressorMixin):
    """
    Just a stupid predictor that takes in some data and returns garbage.
    """

    def __init__(self, minimum=1, maximum=11):
        """
        This is where you would accept the hyper parameters
        """
        self.minimum = minimum
        self.maximum = maximum

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.random.randint(low=self.minimum, high=self.maximum, size=X.shape[0])
