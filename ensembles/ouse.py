from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import math
import warnings
import random
from utils import minority_majority_split, minority_majority_name


class OUSE:

    """
    References
    ----------
    .. [1] Gao, Jing, et al. "Classifying Data Streams with Skewed Class
           Distributions and Concept Drifts." IEEE Internet Computing 12.6
           (2008): 37-49.
    """

    def __init__(self, base_classifier=KNeighborsClassifier(), number_of_classifiers=10, number_of_chunks=10):
        self.base_classifier = base_classifier
        self.number_of_classifiers = number_of_classifiers
        self.classifier_array = []
        self.classifier_weights = []
        self.number_of_chunks = number_of_chunks
        self.minority_name = None
        self.majority_name = None
        self.classes = None
        self.minority_data = []
        self.ratio_chunks = []
        self.label_encoder = None
        self.iterator = 0

    def partial_fit(self, X, y, classes=None):
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)
        if classes is None and self.classes is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y)
            self.classes = self.label_encoder.classes
        elif self.classes is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(classes)
            self.classes = classes

        if classes[0] is "positive":
            self.minority_name = self.label_encoder.transform(classes[0])
            self.majority_name = self.label_encoder.transform(classes[1])
        elif classes[1] is "positive":
            self.minority_name = self.label_encoder.transform(classes[1])
            self.majority_name = self.label_encoder.transform(classes[0])

        y = self.label_encoder.transform(y)

        if self.minority_name is None or self.majority_name is None:
            self.minority_name, self.majority_name = minority_majority_name(y)

        new_minority = self._resample(X, y)
        minority, majority = minority_majority_split(X, y, self.minority_name, self.majority_name)

        if not majority.any():
            print("majoirty empty")
            return

        majority_split = np.array_split(majority, self.number_of_classifiers)

        self.classifier_array = []
        for m_s in majority_split:
            res_X = np.concatenate((m_s, new_minority), axis=0)
            res_y = len(m_s)*[self.majority_name] + len(new_minority)*[self.minority_name]
            new_classifier = self.base_classifier.fit(res_X, res_y)
            self.classifier_array.append(new_classifier)

    def _resample(self, X, y):
        y = np.array(y)
        X = np.array(X)

        minority, majority = minority_majority_split(X, y, self.minority_name, self.majority_name)

        self.minority_data.append(minority.tolist())
        self.ratio_chunks.append(len(minority)/float(len(majority)))
        self.iterator += 1

        if len(self.minority_data) > self.number_of_chunks:
            del self.minority_data[0]
            del self.ratio_chunks[0]

        number_of_instances = len(majority)/self.number_of_classifiers

        new_minority = []
        for md in self.minority_data:
            if number_of_instances < len(md):
                new_minority.extend(random.sample(md, int(number_of_instances)))
            else:
                new_minority.extend(md)

        return new_minority

    def predict(self, X):
        predictions = np.asarray([clf.predict(X) for clf in self.classifier_array]).T
        maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=predictions)
        maj = self.label_encoder.inverse_transform(maj)
        return maj

    def predict_proba(self, X):
        probas_ = [clf.predict_proba(X) for clf in self.classifier_array]
        return np.average(probas_, axis=0)
