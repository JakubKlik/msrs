import sys
import os
from os import listdir
from os.path import isfile, join
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import ranking, ploting, overallScore

streams = []

directory = "gen/sd_b20f5/"
mypath = "streams/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
directory = "gen/sd_b10f5/"
mypath = "streams/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]

directory = "gen/sd_b20f10/"
mypath = "streams/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
directory = "gen/sd_b10f10/"
mypath = "streams/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]

directory = "gen/b20f5/"
mypath = "streams/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
directory = "gen/b10f5/"
mypath = "streams/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]

directory = "gen/b20f10/"
mypath = "streams/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
directory = "gen/b10f10/"
mypath = "streams/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]


directory = "gen/sd_features/"
mypath = "streams/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]

directory = "gen/features/"
mypath = "streams/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]


directory = "gen/sd_balance/"
mypath = "streams/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]

directory = "gen/balance/"
mypath = "streams/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]


streams.sort()
# print(streams)
methods = [
           "MultiSamplingRandomSubspace",
           "KMeanClustering",
           "LearnppCDS",
           "LearnppNIE",
           "REA",
           "OUSE",
           "MLPClassifier",
           ]


# rank = ranking.Ranking(methods, streams, metrics=["balanced_accuracy", "recall", "precision", "cohen_cappa"])
# rank.test_sum(auto_open=False)
for stream_name in streams:
    scr = ploting.Ploting()
    scr.plot(methods, "%s" % stream_name, auto_open=False, metrics=["balanced_accuracy", "recall", "precision", "cohen_cappa"])
# overall = overallScore.OverallScore(methods, streams, metrics=["accuracy", "balanced_accuracy", "recall", "precision", "cohen_cappa"], method_names_alt=["MSRS", "KMC", "L++CDS", "L++NIE", "REA", "OUSE", "MLPC"])
# overall.count_sum()
