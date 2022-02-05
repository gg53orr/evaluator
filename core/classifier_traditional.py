# Created by Andres at 05/02/2022
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from joblib import dump, load


def get_sgd():
    s = SGDClassifier(loss='log', penalty='l2',
                  alpha=1e-3,
                  random_state=42,
                  max_iter=20, tol=None)
    return s


def get_svc():

    linear_svc = LinearSVC()
    return linear_svc


class TraditionalClassifier:
    """
    A trivial classifier
    """
    def __init__(self):
        classifier_algorithm = get_svc()
        self.text_clf = Pipeline([('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
            ('clf', classifier_algorithm)])

    def fit(self, x, y, path):
        """
        Fit
        :param x:
        :param y:
        :param path:
        :return:
        """
        self.text_clf.fit(x, y)
        dump(self.text_clf, path)

    def load(self, path):
        self.text_clf = load()

    def predict_proba(self, a_text: str):
        """
        Prediction and probability
        :param a_text:
        :return:
        """
        predictions = self.text_clf.predict_proba(a_text)
        return predictions

    def predict(self, a_text: str):
        """
        Prediction
        :param a_text:
        :return:
        """
        predictions = self.text_clf.predict(a_text)
        return predictions
