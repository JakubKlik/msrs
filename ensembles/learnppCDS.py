from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import numpy as np
from utils import minority_majority_split, minority_majority_name
import math
import warnings


class LearnppCDS:

    """
    References
    ----------
    .. [1] Ditzler, Gregory, and Robi Polikar. "Incremental learning of
           concept drift from streaming imbalanced data." IEEE Transactions
           on Knowledge and Data Engineering 25.10 (2013): 2283-2301.
    """

    def __init__(self, base_classifier=KNeighborsClassifier(), number_of_classifiers=10, param_a=2, param_b=2):
        self.base_classifier = base_classifier
        self.number_of_classifiers = number_of_classifiers
        self.classifier_array = []
        self.classifier_weights = []
        self.minority_name = None
        self.majority_name = None
        self.classes = None
        self.param_a = param_a
        self.param_b = param_b
        self.label_encoder = None
        self.iterator = 1

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

        y = self.label_encoder.transform(y)

        if self.minority_name is None or self.majority_name is None:
            self.minority_name, self.majority_name = minority_majority_name(y)

        if self.classifier_array:
            y_pred = self.predict(X)
            E = metrics.accuracy_score(y, y_pred)
            w = []
            for i in range(len(y)):
                if y[i] is y_pred[i]:
                    w.append(E/float(len(y)))
                else:
                    w.append(1/float(len(y)))

            D = []
            w_sum = np.sum(w)
            for i in range(len(y)):
                D.append(w[i]/w_sum)

            res_X, res_y = self._resample(X, y)

            new_classifier = self.base_classifier.fit(res_X, res_y)
            self.classifier_array.append(new_classifier)

            epsilon = []
            beta = []
            for j in range(len(self.classifier_array)):
                y_pred = self.classifier_array[j].predict(X)
                for i in range(len(y)):
                    if y[i] is not y_pred[i]:
                        epsilon.append(D[i])
                epsilon_sum = np.sum(epsilon)
                if epsilon_sum > 0.5:
                    if j is len(self.classifier_array) - 1:
                        self.classifier_array[j] = self.base_classifier.fit(res_X, res_y)
                    else:
                        epsilon_sum = 0.5
                beta.append(epsilon_sum / float(1 - epsilon_sum))

            sigma = []
            a = self.param_a
            b = self.param_b
            t = self.iterator
            for k in range(t):
                sigma.append(1/(1 + math.exp(-a*(t-k-b))))

            sigma_mean = []
            for k in range(t):
                sigma_sum = 0
                for j in range(t-k):
                    sigma_sum += sigma[j]
                sigma_mean.append(sigma[k]/sigma_sum)

            beta_mean = []
            for k in range(t):
                beta_sum = 0
                for j in range(t-k):
                    beta_sum += sigma_mean[j]*beta[j]
                beta_mean.append(beta_sum)

            self.classifier_weights = []
            for b in beta_mean:
                self.classifier_weights.append(math.log(1/b))

            self.iterator += 1

        else:
            res_X, res_y = self._resample(X, y)

            new_classifier = self.base_classifier.fit(res_X, res_y)
            self.classifier_array.append(new_classifier)
            self.classifier_weights = [1]
            self.iterator += 1

    def _resample(self, X, y):
        X = np.array(X)
        y = np.array(y)

        minioty, majority = minority_majority_split(X, y, self.minority_name, self.majority_name)
        if len(minioty) > 6:
            res_X, res_y = SMOTE().fit_sample(X, y)
        else:
            res_X, res_y = SMOTE(k_neighbors=len(minioty)-1).fit_sample(X, y)
        return res_X, res_y

    def predict(self, X):
        predictions = np.asarray([clf.predict(X) for clf in self.classifier_array]).T
        maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.classifier_weights)), axis=1, arr=predictions)
        maj = self.label_encoder.inverse_transform(maj)
        return maj

    def predict_proba(self, X):
        probas_ = [clf.predict_proba(X) for clf in self.classifier_array]
        return np.average(probas_, axis=0, weights=self.classifier_weights)
