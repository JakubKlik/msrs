from scipy import stats
import pandas as pd
import os
import numpy as np
import plotly.plotly
import plotly.graph_objs as go
from datetime import datetime


class Ranking:

    def __init__(self, method_names, stream_names, test_name="wilcoxon", metrics=None):
        self.method_names = method_names
        self.stream_names = stream_names
        self.dimension = len(self.method_names)
        self.metrics = metrics
        self.test_name = test_name

        self.date_time = "{:%Y-%m_%d-%H-%M}".format(datetime.now())
        if not os.path.exists("results/ranking_tests/%s/" % (self.date_time)):
            os.makedirs("results/ranking_tests/%s/" % (self.date_time))

    def test_sum(self, treshold=0.05, auto_open=True):
        data = {}
        ranking = {}
        self.iter = 0

        for method_name in self.method_names:
            ranking[method_name] = 0
            for stream_name in self.stream_names:
                try:
                    data[(method_name, stream_name)] = pd.read_csv("results/raw/%s/%s.csv" % (stream_name, method_name), header=0, index_col=0)
                except:
                    print("None is ",method_name, stream_name)
                    data[(method_name, stream_name)] = None

        if self.metrics is None:
            self.metrics = data[(self.method_names[0], self.stream_names[0])].columns.values

        for stream in self.stream_names:
            for metric in self.metrics:
                if self.test_name is "tstudent":
                    for i, method_1 in enumerate(self.method_names):
                        for j, method_2 in enumerate(self.method_names):
                            if method_1 == method_2:
                                continue
                            if data[(method_2, stream_name)] is None:
                                ranking[method_1] += 1
                                print("if1", method_2)
                                self.iter += 1
                                continue
                            if data[(method_1, stream_name)] is None:
                                print("if2", method_1)
                                continue

                            self.iter += 1
                            try:
                                statistic, p_value = stats.ttest_ind(data[(method_1, stream)][metric].values, data[(method_2, stream)][metric].values)
                            except:
                                print(method_1,method_2)
                            if p_value < treshold and statistic > 0:
                                ranking[method_1] += 1
                elif self.test_name is "wilcoxon":
                    for i, method_1 in enumerate(self.method_names):
                        for j, method_2 in enumerate(self.method_names):
                            if method_1 == method_2:
                                continue
                            if data[(method_2, stream_name)] is None:
                                ranking[method_1] += 1
                                print("if1", method_2)
                                self.iter += 1
                                continue
                            if data[(method_1, stream_name)] is None:
                                print("if2", method_1)
                                continue

                            self.iter += 1
                            try:
                                statistic, p_value = stats.ranksums(data[(method_1, stream)][metric].values, data[(method_2, stream)][metric].values)
                                if p_value < treshold:
                                    ranking[method_1] += statistic
                            except:
                                print(method_1,method_2)

        trace = self.prepare_trace(ranking)
        layout = go.Layout(title='Ranking %s tests summarise' % (self.test_name), plot_bgcolor='rgb(230, 230, 230)')
        fig = dict(data=[trace], layout=layout)
        plotly.offline.plot(fig, filename="results/ranking_tests/%s/ranking_sum_%s.html" % (self.date_time, self.test_name), auto_open=auto_open)

    def test_streams(self, treshold=0.001, auto_open=True):
        data = {}
        ranking = {}
        for method_name in self.method_names:
            ranking[method_name] = 0
            for stream_name in self.stream_names:
                data[(method_name, stream_name)] = pd.read_csv("results/raw/%s/%s.csv" % (stream_name, method_name), header=0, index_col=0)

        if not self.metrics:
            self.metrics = data[(self.method_names[0], self.stream_names[0])].columns.values

        for stream in self.stream_names:
            ranking = {}
            self.iter = 0
            for method_name in self.method_names:
                ranking[method_name] = 0
            for metric in self.metrics:
                if self.test_name is "tstudent":
                    for i, method_1 in enumerate(self.method_names):
                        for j, method_2 in enumerate(self.method_names):
                            self.iter += 1
                            statistic, p_value = stats.ttest_ind(data[(method_1, stream)][metric].values, data[(method_2, stream)][metric].values)
                            if p_value < treshold and statistic > 0:
                                ranking[method_1] += 1

            trace = self.prepare_trace(ranking)
            stream = stream.split("/")[1]
            layout = go.Layout(title='Ranking %s tests for %s' % (self.test_name, stream), plot_bgcolor='rgb(230, 230, 230)')
            fig = dict(data=[trace], layout=layout)
            plotly.offline.plot(fig, filename="results/ranking_tests/%s_%s/ranking_%s.html" % (self.date_time, stream, self.test_name), auto_open=auto_open)

    def test_metrics(self, treshold=0.001, auto_open=True):
        data = {}
        ranking = {}
        for method_name in self.method_names:
            ranking[method_name] = 0
            for stream_name in self.stream_names:
                data[(method_name, stream_name)] = pd.read_csv("results/raw/%s/%s.csv" % (stream_name, method_name), header=0, index_col=0)

        if not self.metrics:
            self.metrics = data[(self.method_names[0], self.stream_names[0])].columns.values

        for metric in self.metrics:
            ranking = {}
            self.iter = 0
            for method_name in self.method_names:
                ranking[method_name] = 0
            for stream in self.stream_names:
                if self.test_name is "tstudent":
                    for i, method_1 in enumerate(self.method_names):
                        for j, method_2 in enumerate(self.method_names):
                            self.iter += 1
                            statistic, p_value = stats.ttest_ind(data[(method_1, stream)][metric].values, data[(method_2, stream)][metric].values)
                            if p_value < treshold and statistic > 0:
                                ranking[method_1] += 1
                elif self.test_name is "wilcoxon":
                    for i, method_1 in enumerate(self.method_names):
                        for j, method_2 in enumerate(self.method_names):
                            statistic, p_value = stats.wilcoxon(data[(method_1, stream)][metric].values, data[(method_2, stream)][metric].values)
                            if p_value < treshold and statistic > 0:
                                ranking[method_1] += 1

            trace = self.prepare_trace(ranking)
            layout = go.Layout(title='Ranking %s tests for %s' % (self.test_name, metric), plot_bgcolor='rgb(230, 230, 230)')
            fig = dict(data=[trace], layout=layout)
            plotly.offline.plot(fig, filename="results/ranking_tests/%s_%s/ranking_%s.html" % (self.date_time, metric, self.test_name), auto_open=auto_open)

    def prepare_trace(self, ranking):
        items = ranking.items()

        vals = []
        names = []

        for it in sorted(items, key=lambda item: item[1], reverse=True):
            names.append(it[0])
            # vals.append(it[1])
            vals.append(round(it[1]/float(self.iter)*1000, 2))

        trace = go.Table(
            columnwidth=[20, 50, 50],
            header=dict(values=["<b>Position<b>", "<b>Method<b>", "<b>Score<b>"],
                        line=dict(color='#506784'),
                        fill=dict(color='#119DFF'),
                        align=['center'],
                        font=dict(color='white', size=16),
                        height=32),
            cells=dict(values=[list(range(1, len(names)+1)), names, vals],
                       line=dict(color='#506784'),
                       fill=dict(color=['white']),
                       align=['center'],
                       font=dict(color='#506784', size=16),
                       height=32
                       ))
        return trace
