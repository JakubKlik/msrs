import pandas as pd
import plotly.graph_objs as go
import plotly.plotly
import os


class Ploting:
    def __init__(self, directory="./"):
        self.directory = directory

    def plot(self, method_names, stream_name, auto_open=True, metrics=None):
        self.method_names = method_names
        self.stream_name = stream_name
        self.metrics = metrics

        data_trace = None
        if not os.path.exists("results/plots/%s/" % self.stream_name):
            os.makedirs("results/plots/%s/" % self.stream_name)

        for method_name in self.method_names:
            try:
                data = pd.read_csv("results/raw/%s/%s.csv" % (self.stream_name, method_name), header=0, index_col=0)

                if data_trace is None:
                    if self.metrics is not None:
                        data_trace = [[] for column_name in self.metrics]
                        column_names = self.metrics
                    else:
                        data_trace = [[] for column_name in data.columns]
                        column_names = data.columns

                for i, column_name in enumerate(column_names):
                    data_trace[i] += [go.Scatter(
                        x=data.index.values, y=data[column_name].values, name='%s' % method_name,
                        mode='lines',
                        line=dict(width=1),
                        showlegend=True
                    )]
            except(FileNotFoundError):
                continue

        for i, column_name in enumerate(column_names):
            layout = go.Layout(title='%s, data stream - %s' % (column_name, self.stream_name), plot_bgcolor='rgb(230, 230, 230)')
            fig = go.Figure(data=data_trace[i], layout=layout)
            plotly.offline.plot(fig, filename="results/plots/%s/%s.html" % (self.stream_name,column_name), auto_open=auto_open)

    def plot_streams(self, streams, method_name, auto_open=True):

        data_trace = None
        if not os.path.exists("results/plots/%s/" % method_name):
            os.makedirs("results/plots/%s/" % method_name)

        for stream in streams:
            try:
                stream = self.directory + stream
                data = pd.read_csv("results/raw/%s/%s.csv" % (stream, method_name), header=0, index_col=0)

                if data_trace is None:
                    data_trace = [[] for column_name in data.columns]

                for i, column_name in enumerate(data.columns):
                    data_trace[i] += [go.Scatter(
                        x=data.index.values, y=data[column_name].values, name='%s' % stream,
                        mode='lines',
                        line=dict(width=1),
                        showlegend=True
                    )]
            except(FileNotFoundError):
                continue

        for i, column_name in enumerate(data.columns):
            layout = go.Layout(title='%s, method - %s' % (column_name, method_name), plot_bgcolor='rgb(230, 230, 230)')
            fig = go.Figure(data=data_trace[i], layout=layout)
            plotly.offline.plot(fig, filename="results/plots/%s/%s.html" % (method_name, column_name), auto_open=auto_open)
