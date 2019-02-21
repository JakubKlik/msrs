import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from utils import minority_majority_split, minority_majority_name
from imblearn import under_sampling
import random
import warnings
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek

class MultiSamplingRandomSubspace():

    def __init__(self, base_classifier=KNeighborsClassifier(), number_of_classifiers=10, weight_method=metrics.balanced_accuracy_score, sampling_methods=[CondensedNearestNeighbour, TomekLinks, ADASYN, SMOTE, SMOTEENN, SMOTETomek]):
        self.base_classifier = base_classifier
        self.number_of_classifiers = number_of_classifiers
        self.weight_method = weight_method
        self.sampling_methods = sampling_methods

        self.subspace_array = []
        self.classifier_array = []
        self.classifier_weights = []
        self.classes = None
        self.label_encoder = None
        self.minority_name = None
        self.majority_name = None
        self.iterator = 0
        self.sms = []

    def partial_fit(self, X, y, classes):
        if self.classes is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(classes)
            self.classes = classes

        y = self.label_encoder.transform(y)

        if self.minority_name is None or self.majority_name is None:
            self.minority_name, self.majority_name = minority_majority_name(y)

        subspace = random.sample(range(len(X[0])), int(len(X[0]/2)))
        sampling_methods_pool = self.sampling_methods

        while(True):
            try:
                sampling_method = random.choice(sampling_methods_pool)
                new_clf = self._new_classifier(X, y, sampling_method, subspace)
                break
            except:
                sampling_methods_pool.remove(sampling_method)
                continue

        if len(self.classifier_array) == self.number_of_classifiers:
            remove_index = self.classifier_weights.index(min(self.classifier_weights))
            del self.classifier_weights[remove_index]
            del self.classifier_array[remove_index]
            del self.subspace_array[remove_index]
            del self.sms[remove_index]

        self.subspace_array.append(subspace)
        self.classifier_array.append(new_clf)
        self.sms.append(sampling_method.__name__)

        new_weigths = []
        for clf, sub in zip(self.classifier_array, self.subspace_array):
            y_pred = clf.predict(X[:,sub])
            new_weigths.append(self.weight_method(y, y_pred))
            self.classifier_weights = new_weigths

        # print(self.sms)

    def predict(self, X):
        predictions = []
        for clf, sub in zip(self.classifier_array, self.subspace_array):
            predictions.append(clf.predict(X[:,sub]))
        predictions = np.array(predictions).T
        maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.classifier_weights)), axis=1, arr=predictions)
        maj = self.label_encoder.inverse_transform(maj)
        return maj

    def predict_proba(self, X):
        probas_ = []
        for clf, sub in zip(self.classifier_array, self.subspace_array):
            probas_.append(clf.predict_proba(X[:,sub]))
        return np.average(probas_, axis=0, weights=self.classifier_weights)

    def _resample(self, X, y, sampling_method):
        y = np.array(y)
        X = np.array(X)

        res_X, res_y = sampling_method().fit_sample(X, y)

        return res_X, res_y

    def _new_classifier(self, X, y, sampling_method, subspace):
        y = np.array(y)
        X = np.array(X)

        res_X, res_y = self._resample(X, y, sampling_method)

        return self.base_classifier.fit(res_X[:,subspace], res_y)
