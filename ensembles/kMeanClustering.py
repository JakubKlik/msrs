from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
from utils import minority_majority_split, minority_majority_name
import warnings


class KMeanClustering:

    """
    References
    ----------
    .. [1] Wang, Yi, Yang Zhang, and Yong Wang. "Mining data streams
           with skewed distribution by static classifier ensemble."
           Opportunities and Challenges for Next-Generation Applied
           Intelligence. Springer, Berlin, Heidelberg, 2009. 65-71.
    """

    def __init__(self, base_classifier=KNeighborsClassifier(), number_of_classifiers=10):
        self.base_classifier = base_classifier
        self.number_of_classifiers = number_of_classifiers
        self.classifier_array = []
        self.classifier_weights = []
        self.minority_name = None
        self.majority_name = None
        self.classes = None
        self.label_encoder = None

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

        res_X, res_y = self._resample(X, y)
        if res_X is None:
            return
        new_classifier = self.base_classifier.fit(res_X, res_y)

        if len(self.classifier_array) < self.number_of_classifiers:
            self.classifier_array.append(new_classifier)
            self.classifier_weights.append(1)
        else:
            auc_array = []
            for i in range(len(self.classifier_array)):
                y_score = self.classifier_array[i].predict_proba(res_X)
                fpr, tpr, thresholds = metrics.roc_curve(res_y, y_score[:,0])
                auc_array += [metrics.auc(fpr, tpr)]

            j = np.argmin(auc_array)

            y_score = new_classifier.predict_proba(res_X)
            fpr, tpr, thresholds = metrics.roc_curve(res_y, y_score[:,0])
            new_auc = metrics.auc(fpr, tpr)

            if new_auc > auc_array[j]:
                self.classifier_array[j] = new_classifier
                auc_array[j] = new_auc

            # auc_norm = auc_array / np.linalg.norm(auc_array)
            for i in range(len(self.classifier_array)):
                self.classifier_weights[i] = auc_array[i]

    def _resample(self, X, y):
        y = np.array(y)
        X = np.array(X)

        minority, majority = minority_majority_split(X, y, self.minority_name, self.majority_name)

        # Undersample majority array
        if len(minority) != 0:
            km = KMeans(n_clusters=len(minority)).fit(X)
            majority = km.cluster_centers_

            res_X = np.concatenate((majority, minority), axis=0)
            res_y = len(majority)*[self.majority_name] + len(minority)*[self.minority_name]

            return res_X, res_y
        else:
            return None, None

    def predict(self, X):
        predictions = np.asarray([clf.predict(X) for clf in self.classifier_array]).T
        maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.classifier_weights)), axis=1, arr=predictions)
        maj = self.label_encoder.inverse_transform(maj)
        return maj

    def predict_proba(self, X):
        probas_ = [clf.predict_proba(X) for clf in self.classifier_array]
        return np.average(probas_, axis=0, weights=self.classifier_weights)
