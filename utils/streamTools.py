import numpy as np
import plotly.plotly
import plotly.figure_factory as ff
import pandas as pd


def preview_stream(data, stream_name):
    data = list(list(i) for i in data)
    objects = np.asarray(data)
    dataframe = pd.DataFrame(objects[:, 0:-1])

    dataframe['Classes'] = pd.Series(objects[:, -1])
    fig = ff.create_scatterplotmatrix(dataframe, diag='box', index='Classes', title='Preview data stream %s. Ratio = %2f, Objects = %d' % (stream_name, check_percentage(data), len(data)), height=1800, width=1800)

    plotly.offline.plot(fig, filename="plots/preview_%s.html" % stream_name)


def __float_cast(item):
    try:
        return float(item)
    except ValueError:
        return item


def check_percentage(data):
    data = list(list(i) for i in data)
    objects = np.asarray(data)
    uniq = np.unique(objects[:, -1])
    positive = objects[objects[:, -1] == uniq[0]]
    negative = objects[objects[:, -1] == uniq[1]]
    positive = np.asarray(positive)

    ratio = []

    for i in range(len(data[0])-1):
        min_p = float(min(positive[:, i]))
        max_p = float(max(positive[:, i]))
        min_n = float(min(negative[:, i]))
        max_n = float(max(negative[:, i]))
        if min_p < min_n:
            if max_p < max_n and min_p != max_n:
                ratio.append((max_p - min_n) / float(max_n - min_p))
            elif min_p != max_p and min_n != max_n:
                ratio.append((max_n - min_n) / float(max_p - min_p))
        else:
            if max_p > max_n and min_p != max_n:
                ratio.append((min_p - max_n) / float(min_n - max_p))
            elif min_p != max_p and min_n != max_n:
                ratio.append((min_p - max_p) / float(min_n - max_n))
    average_ratio = 1
    for r in ratio:
        r = abs(r)
        average_ratio *= r

    return average_ratio


def __prepareDataDF(data):
    df = pd.DataFrame(data)
    features = df.iloc[:, 0:-1].values.astype(float)
    labels = df.iloc[:, -1].values.astype(str)
    classes = np.unique(labels)
    return features, labels, classes


def __prepareDataNumpy(data):
    features = np.delete(data, -1, axis=1)
    features = features.astype(float)
    labels = data[:, -1].astype(str)
    classes = np.unique(labels)
    return features, labels, classes


def __prepareDataLoop(data):
    features, labels = [], []
    for raw_object in data:
        prepared_object = [__float_cast(item) for item in raw_object]
        prepared_object = prepared_object[:-1]
        features.append(prepared_object)
        className = raw_object[-1]
        labels.append(className)

    classes = np.unique(labels)
    return features, labels, classes


def prepareData(data):
    try:
        return __prepareDataDF(data)
    except ValueError:
        return __prepareDataLoop(data)
