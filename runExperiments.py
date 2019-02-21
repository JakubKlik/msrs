import os
from os import listdir
from os.path import isfile, join
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ensembles import KMeanClustering
from ensembles import LearnppCDS
from ensembles import LearnppNIE
from ensembles import REA
from ensembles import OUSE
from ensembles import MultiSamplingRandomSubspace
from sklearn.neural_network import MLPClassifier

from utils import ranking
from utils import evaluation
from utils import streamTools
from utils import ploting
from scipy.io import arff

from joblib import Parallel, delayed
import time

import warnings
import traceback
warnings.simplefilter("ignore")

def load_data(stream_name):
    data, meta = arff.loadarff("streams/%s.arff" % stream_name)
    classes = meta[meta.names()[-1]][1]
    return data, classes

def evaluate_method(classifier, stream_name, method_name, initial_size, step_size):
    print(stream_name, method_name)
    try:
        start = time.time()
        data, meta = arff.loadarff("streams/%s.arff" % stream_name)
        if data is None:
            print("Empty data")
            raise Exception
        classes = meta[meta.names()[-1]][1]
        evl = evaluation.Evaluation(classifier=classifier, stream_name="%s" % stream_name, method_name=method_name, tqdm=False)
        evl.test_and_train(data=data, classes=classes, initial_size=initial_size, step_size=step_size)
        evl.compute_metrics()
        evl.save_to_csv_metrics()
        print("End", stream_name, method_name, time.time()-start)
    except Exception as ex:
        print(str(ex))
        traceback.print_exc()
        print("Exception in ", stream_name, method_name)

cores = open('/proc/cpuinfo').read().count('processor\t:')


methods = [
           KMeanClustering(),
           LearnppCDS(),
           LearnppNIE(),
           REA(),
           OUSE(),
           MultiSamplingRandomSubspace(),
           MLPClassifier(),
           ]


step_size = 500
initial_size = 2*step_size

streams=["gen/stream_gen_20k_0.10_05_f5_normal"]
print("Start")
start = time.time()

Parallel(n_jobs=-1)(
    delayed(evaluate_method)(classifier, stream_name, type(classifier).__name__, initial_size, step_size)
        for classifier in methods for stream_name in streams)

end = time.time()
print("End %f" % (end-start))



######################################
# Features
######################################

# directory = "gen/features/"
# mypath = "streams/%s" % directory
# streams = ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
#
# print("Start", directory)
# start = time.time()
#
# Parallel(n_jobs=-1)(
#     delayed(evaluate_method)(classifier, stream_name, type(classifier).__name__, initial_size, step_size)
#         for classifier in methods for stream_name in streams)
#
# end = time.time()
# print("End %f" % (end-start))
#
#
# directory = "gen/sd_features/"
# mypath = "streams/%s" % directory
# streams = ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
#
#
# print("Start", directory)
# start = time.time()
#
# Parallel(n_jobs=-1)(
#     delayed(evaluate_method)(classifier, stream_name, type(classifier).__name__, initial_size, step_size)
#         for classifier in methods for stream_name in streams)
#
# end = time.time()
# print("End %f" % (end-start))
#
# ######################################
# # Balance
# ######################################
#
# directory = "gen/balance/"
# mypath = "streams/%s" % directory
# streams = ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
#
# print("Start", directory)
# start = time.time()
#
# Parallel(n_jobs=-1)(
#     delayed(evaluate_method)(classifier, stream_name, type(classifier).__name__, initial_size, step_size)
#         for classifier in methods for stream_name in streams)
#
# end = time.time()
# print("End %f" % (end-start))
#
#
# directory = "gen/sd_balance/"
# mypath = "streams/%s" % directory
# streams = ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
#
#
# print("Start", directory)
# start = time.time()
#
# Parallel(n_jobs=-1)(
#     delayed(evaluate_method)(classifier, stream_name, type(classifier).__name__, initial_size, step_size)
#         for classifier in methods for stream_name in streams)
#
# end = time.time()
# print("End %f" % (end-start))
#
# ######################################
# # Stream no-drift
# ######################################
#
# directory = "gen/b10f5/"
# mypath = "streams/%s" % directory
# streams = ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
#
# print("Start", directory)
# start = time.time()
#
# Parallel(n_jobs=-1)(
#     delayed(evaluate_method)(classifier, stream_name, type(classifier).__name__, initial_size, step_size)
#         for classifier in methods for stream_name in streams)
#
# end = time.time()
# print("End %f" % (end-start))
#
# ################################
#
# directory = "gen/b10f10/"
# mypath = "streams/%s" % directory
# streams = ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
#
#
# print("Start", directory)
# start = time.time()
#
# Parallel(n_jobs=-1)(
#     delayed(evaluate_method)(classifier, stream_name, type(classifier).__name__, initial_size, step_size)
#         for classifier in methods for stream_name in streams)
#
# end = time.time()
# print("End %f" % (end-start))
#
# ######################################
#
# directory = "gen/b20f5/"
# mypath = "streams/%s" % directory
# streams = ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
#
# print("Start", directory)
# start = time.time()
#
# Parallel(n_jobs=-1)(
#     delayed(evaluate_method)(classifier, stream_name, type(classifier).__name__, initial_size, step_size)
#         for classifier in methods for stream_name in streams)
#
# end = time.time()
# print("End %f" % (end-start))
#
# ################################
#
# directory = "gen/b20f10/"
# mypath = "streams/%s" % directory
# streams = ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
#
#
# print("Start", directory)
# start = time.time()
#
# Parallel(n_jobs=-1)(
#     delayed(evaluate_method)(classifier, stream_name, type(classifier).__name__, initial_size, step_size)
#         for classifier in methods for stream_name in streams)
#
# end = time.time()
# print("End %f" % (end-start))
#
# ######################################
#
# directory = "gen/b30f5/"
# mypath = "streams/%s" % directory
# streams = ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
#
# print("Start", directory)
# start = time.time()
#
# Parallel(n_jobs=-1)(
#     delayed(evaluate_method)(classifier, stream_name, type(classifier).__name__, initial_size, step_size)
#         for classifier in methods for stream_name in streams)
#
# end = time.time()
# print("End %f" % (end-start))
#
# ################################
#
# directory = "gen/b30f10/"
# mypath = "streams/%s" % directory
# streams = ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
#
#
# print("Start", directory)
# start = time.time()
#
# Parallel(n_jobs=-1)(
#     delayed(evaluate_method)(classifier, stream_name, type(classifier).__name__, initial_size, step_size)
#         for classifier in methods for stream_name in streams)
#
# end = time.time()
# print("End %f" % (end-start))
#
# ######################################
# # Stream sudden drift
# ######################################
#
# directory = "gen/sd_b10f5/"
# mypath = "streams/%s" % directory
# streams = ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
#
# print("Start", directory)
# start = time.time()
#
# Parallel(n_jobs=-1)(
#     delayed(evaluate_method)(classifier, stream_name, type(classifier).__name__, initial_size, step_size)
#         for classifier in methods for stream_name in streams)
#
# end = time.time()
# print("End %f" % (end-start))
#
# ################################
#
# directory = "gen/sd_b10f10/"
# mypath = "streams/%s" % directory
# streams = ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
#
#
# print("Start", directory)
# start = time.time()
#
# Parallel(n_jobs=-1)(
#     delayed(evaluate_method)(classifier, stream_name, type(classifier).__name__, initial_size, step_size)
#         for classifier in methods for stream_name in streams)
#
# end = time.time()
# print("End %f" % (end-start))
#
# ######################################
#
# directory = "gen/sd_b20f5/"
# mypath = "streams/%s" % directory
# streams = ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
#
# print("Start", directory)
# start = time.time()
#
# Parallel(n_jobs=-1)(
#     delayed(evaluate_method)(classifier, stream_name, type(classifier).__name__, initial_size, step_size)
#         for classifier in methods for stream_name in streams)
#
# end = time.time()
# print("End %f" % (end-start))
#
# ################################
#
# directory = "gen/sd_b20f10/"
# mypath = "streams/%s" % directory
# streams = ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
#
#
# print("Start", directory)
# start = time.time()
#
# Parallel(n_jobs=-1)(
#     delayed(evaluate_method)(classifier, stream_name, type(classifier).__name__, initial_size, step_size)
#         for classifier in methods for stream_name in streams)
#
# end = time.time()
# print("End %f" % (end-start))
#
# ######################################
#
# directory = "gen/sd_b30f5/"
# mypath = "streams/%s" % directory
# streams = ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
#
# print("Start", directory)
# start = time.time()
#
# Parallel(n_jobs=-1)(
#     delayed(evaluate_method)(classifier, stream_name, type(classifier).__name__, initial_size, step_size)
#         for classifier in methods for stream_name in streams)
#
# end = time.time()
# print("End %f" % (end-start))
#
# ################################
#
# directory = "gen/sd_b30f10/"
# mypath = "streams/%s" % directory
# streams = ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
#
#
# print("Start", directory)
# start = time.time()
#
# Parallel(n_jobs=-1)(
#     delayed(evaluate_method)(classifier, stream_name, type(classifier).__name__, initial_size, step_size)
#         for classifier in methods for stream_name in streams)
#
# end = time.time()
# print("End %f" % (end-start))
