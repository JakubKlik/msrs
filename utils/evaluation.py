from . import streamTools as st
import csv
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import os
import threading

# threadLock = threading.Lock()
import warnings
warnings.simplefilter("ignore")

class Evaluation(threading.Thread):

    def __init__(self, classifier, stream_name, method_name, tqdm=True):
        self.__classifier = classifier
        self.__stream_name = stream_name
        self.__method_name = method_name
        self.__tqdm = tqdm

        self.__data = None
        self.__classes = None
        self.__X = []
        self.__true_y = []
        self.__predict_y = []
        self.__predict_probas = []
        self.__balanced_accuracy = []
        self.__auc = []
        self.__recall = []
        self.__cohen_cappa = []
        self.__matthews_corrcoef = []
        self.__cappa_m = []
        self.__harmonic_mean = []
        self.__fbeta_score = []
        self.__metric_names = []
        self.__step_size = None
        threading.Thread.__init__(self)

    def run(self):
        # print "Start %s \n" % self.__method_name
        threadLock.acquire()
        self.test_and_train(self.__data)
        self.compute_metrics()
        self.save_to_csv_metrics()
        threadLock.release()
        print("Done %s \n" % self.__method_name)

    def test_and_train(self, data, classes, steps=20, initial_steps=1, initial_size=None, step_size=None, online=False):

        # print "test_and_train_Start \n"
        if step_size is None:
            self.__step_size = len(data)/steps
            self.__steps = steps
        else:
            self.__step_size = step_size
            self.__steps = int(len(data)/step_size)

        if initial_size is None or initial_size == 0:
            self.__initial_size = self.__step_size*initial_steps
            self.__initial_steps = initial_steps
        else:
            self.__initial_size = initial_size
            self.__initial_steps = int(initial_size/self.__step_size)

        initial_data = data[0:self.__initial_size]
        X, y, classes_ = st.prepareData(initial_data)
        self.__classes = classes

        if online:
            self.__classifier.fit(X, y, self.__classes)

            for i in tqdm(range(self.__initial_size, len(data)), desc=self.__method_name):
                X, y, c = st.prepareData(data[i:(i+1)])

                predict = self.__classifier.predict(X)
                self.__classifier.partial_fit(X, y, self.__classes)

                # if (i+1) % self.__step_size is 0 and i != self.__initial_size:
                self.__gather_data(X, y, predict)

            return self.__classifier

        else:
            self.__classifier.partial_fit(X, y, self.__classes)

            if(self.__tqdm):
                for i in tqdm(range(self.__initial_steps, self.__steps), desc=self.__method_name):
                    chunk = data[(i*self.__step_size):((i+1)*self.__step_size)]
                    X, y, c = st.prepareData(chunk)

                    predict = self.__classifier.predict(X)
                    self.__classifier.partial_fit(X, y, self.__classes)
                    self.__gather_data(X, y, predict)
            else:
                for i in range(self.__initial_steps, self.__steps):
                    chunk = data[(i*self.__step_size):((i+1)*self.__step_size)]
                    X, y, c = st.prepareData(chunk)

                    predict = self.__classifier.predict(X)
                    self.__classifier.partial_fit(X, y, self.__classes)
                    self.__gather_data(X, y, predict)

            return self.__classifier

    def __gather_data(self, X, y, predict):
        self.__X.extend(X)
        self.__true_y.extend(y)
        self.__predict_y.extend(predict)
        self.__predict_probas.extend(self.__classifier.predict_proba(X))

    def compute_metrics(self):
        step = len(self.__X) / self.__step_size
        self.__metric_names = ",accuracy,balanced_accuracy,recall,precision,cohen_cappa,cappa_m,matthews_corrcoef,harmonic_mean,fbeta_score,auc,f1_score"
        # self.__metric_names = ",balanced_accuracy,auc,recall,cohen_cappa,matthews_corrcoef,cappa_m,harmonic_mean,fbeta_score"

        accuracy = []
        balanced_accuracy = []
        recall = []
        precision = []
        cohen_cappa = []
        cappa_m = []
        auc = []
        f1_score = []
        matthews_corrcoef = []
        harmonic_mean = []
        fbeta_score = []

        y_true_bin = label_binarize(self.__true_y, np.unique(self.__true_y))
        y_pred_bin = label_binarize(self.__predict_y, np.unique(self.__predict_y))
        for i in range(0, int(step)):
            y_true = y_true_bin[i*self.__step_size:(i+1)*self.__step_size]
            y_pred = y_pred_bin[i*self.__step_size:(i+1)*self.__step_size]
            predict_probas = np.array(self.__predict_probas[i*self.__step_size:(i+1)*self.__step_size])

            accuracy += [metrics.accuracy_score(y_true, y_pred)]
            balanced_accuracy += [metrics.balanced_accuracy_score(y_true, y_pred)]
            recall += [metrics.recall_score(y_true, y_pred)]
            precision += [metrics.precision_score(y_true, y_pred)]
            cohen_cappa += [metrics.cohen_kappa_score(y_true, y_pred)]
            cappa_m += [self.cappa_m(y_true, y_pred)]
            f1_score += [metrics.f1_score(y_true, y_pred)]
            matthews_corrcoef += [metrics.matthews_corrcoef(y_true, y_pred)]
            harmonic_mean += [self.harmonic_mean(y_true, y_pred)]
            fbeta_score += [metrics.fbeta_score(y_true, y_pred, 0.5)]
            try:
                auc += [metrics.roc_auc_score(y_true, predict_probas[:,0])]
            except ValueError:
                auc += [0]

        self.__accuracy = np.array(accuracy)
        self.__balanced_accuracy = np.array(balanced_accuracy)
        self.__recall = np.array(recall)
        self.__precision = np.array(precision)
        self.__cohen_cappa = np.array(cohen_cappa)
        self.__cappa_m = np.array(cappa_m)
        self.__auc = np.array(auc)
        self.__f1_score = np.array(f1_score)
        self.__matthews_corrcoef = np.array(matthews_corrcoef)
        self.__harmonic_mean = np.array(harmonic_mean)
        self.__fbeta_score = np.array(fbeta_score)

    def cappa_m(self, y1, y2):
        try:
            tn, fp, fn, tp = metrics.confusion_matrix(y1, y2).ravel()
        except ValueError:
            return 0

        p_0 = (tn + tp) / float(len(y1))
        p_m = (fp + tn) / float(len(y1))

        return (p_0 - p_m) / float(1 - p_m)

    def harmonic_mean(self, y1, y2):
        tn, fp, fn, tp = metrics.confusion_matrix(y1, y2).ravel()

        if tp != 0:
            s1 = 1 / (tp / float(tp+fn))
        else:
            s1 = 1

        if tn != 0:
            s2 = 1 / (tn / float(tn+fp))
        else:
            s2 = 1

        return 2 / s1 + s2

    def save_to_csv(self, filename):
        with open(filename, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for i in range(len(self.__X)):
                for j in range(len(self.__X[i])):
                    writer.writerow([
                        '%i' % i,
                        '%s' % self.__true_y[i][j],
                        '%s' % self.__predict_y[i][j],
                        '%f' % self.__predict_probas[i][j][0],
                        '%f' % self.__predict_probas[i][j][1]
                    ])

    def save_to_csv_metrics(self, filename=None):
        if filename is None:
            filename = "results/raw/%s/%s.csv" % (self.__stream_name, self.__method_name)

        if not os.path.exists("results/raw/%s/" % self.__stream_name):
            os.makedirs("results/raw/%s/" % self.__stream_name)

        # index = range(len(self.__X)/self.__step_size)
        data = np.stack((range(int(len(self.__X)/self.__step_size)),
                         self.__accuracy,
                         self.__balanced_accuracy,
                         self.__recall,
                         self.__precision,
                         self.__cohen_cappa,
                         self.__cappa_m,
                         self.__matthews_corrcoef,
                         self.__harmonic_mean,
                         self.__fbeta_score,
                         self.__auc,
                         self.__f1_score), axis=-1)

        np.savetxt(fname=filename,
                   fmt="%d, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f",
                   header=self.__metric_names,
                   X=data)
