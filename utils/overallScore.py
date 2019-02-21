from scipy import stats
import pandas as pd
import os
import numpy as np
from datetime import datetime

class OverallScore:
    def __init__(self, method_names, stream_names, metrics=None, method_names_alt=None):
        self.method_names = method_names
        self.stream_names = stream_names
        self.dimension = len(self.method_names)
        self.metrics = metrics
        self.method_names_alt = method_names_alt

    def count(self, filename=None):
        data = {}
        ranking = {}
        self.iter = 0

        self.date_time = "{:%Y-%m_%d-%H-%M}".format(datetime.now())
        if not os.path.exists("results/overal_score/%s/" % (self.date_time)):
            os.makedirs("results/overal_score/%s/" % (self.date_time))


        for method_name in self.method_names:
            ranking[method_name] = 0
            for stream_name in self.stream_names:
                data[(method_name, stream_name)] = pd.read_csv("results/raw/%s/%s.csv" % (stream_name, method_name), header=0, index_col=0)

        if not self.metrics:
            self.metrics = data[(self.method_names[0], self.stream_names[0])].columns.values

        for metric in self.metrics:
            # df = pd.DataFrame()
            dict_data = {}
            dict_mean = {}
            dict_std = {}
            for method in self.method_names:
                stream_array = []
                stream_mean = []
                stream_std = []
                for stream in self.stream_names:
                    temp_data = data[(method, stream)][metric].values
                    stream_mean.append(np.mean(temp_data))
                    # stream_std.append(np.std(temp_data))
                dict_data["%s_mean" % method] = stream_mean
                # dict_data["%s_std" % method] = stream_std
            column_names = []
            # if self.method_names_alt is None:
            for mn in self.method_names:
                column_names.append("%s_mean" % mn)
                # column_names.append("%s_std" % mn)
            df = pd.DataFrame.from_records(dict_data, index=self.stream_names, columns=column_names)
            # else:
            #     df = pd.DataFrame.from_records(dict_data, index=self.stream_names, columns=self.method_names_alt)

            if not filename:
                filename = "count"
            df.to_csv("results/overal_score/%s/%s_%s.csv" % (self.date_time, metric, filename))

    def count_latex(self):
        data = {}
        ranking = {}
        self.iter = 0

        self.date_time = "{:%Y-%m_%d-%H-%M}".format(datetime.now())
        if not os.path.exists("results/overal_score/%s/" % (self.date_time)):
            os.makedirs("results/overal_score/%s/" % (self.date_time))


        for method_name in self.method_names:
            ranking[method_name] = 0
            for stream_name in self.stream_names:
                data[(method_name, stream_name)] = pd.read_csv("results/raw/%s/%s.csv" % (stream_name, method_name), header=0, index_col=0)

        if not self.metrics:
            self.metrics = data[(self.method_names[0], self.stream_names[0])].columns.values

        for metric in self.metrics:
            file_text = [ "" for i in range(len(self.stream_names)+1)]
            file_text[0] = "Stream difficulty"
            dict_mean = {}
            dict_std = {}
            for i, stream in enumerate(self.stream_names):
                file_text[i+1] = stream.split("_")[-3]

            for i, method in enumerate(self.method_names):
                file_text[0] += ","
                if self.method_names_alt is None:
                    file_text[0] += method
                else:
                    file_text[0] += self.method_names_alt[i]
                stream_mean = []
                stream_std = []
                for j, stream in enumerate(self.stream_names):

                    temp_data = data[(method, stream)][metric].values
                    mean = np.mean(temp_data)
                    std = np.std(temp_data)
                    file_text[j+1] += r',%0.2f\pm%0.2f' % (mean*100,std*100)

                    stream_mean.append(np.mean(temp_data))
                    stream_std.append(np.std(temp_data))
                    dict_mean[method] = stream_mean
                    dict_std[method] = stream_std

                print(metric, method,np.mean(dict_mean[method]))

    def count_sum(self, filename=None):
        data = {}
        ranking = {}
        self.iter = 0

        self.date_time = "{:%Y-%m_%d-%H-%M}".format(datetime.now())
        if not os.path.exists("results/overal_score/%s/" % (self.date_time)):
            os.makedirs("results/overal_score/%s/" % (self.date_time))

        for method_name in self.method_names:
            ranking[method_name] = 0
            for stream_name in self.stream_names:
                data[(method_name, stream_name)] = pd.read_csv("results/raw/%s/%s.csv" % (stream_name, method_name), header=0, index_col=0)

        if not self.metrics:
            self.metrics = data[(self.method_names[0], self.stream_names[0])].columns.values

        file_text = [ "" for i in range(len(self.method_names)+1)]
        file_text[0] = "method"

        if self.method_names_alt is None:
            for i, method in enumerate(self.method_names):
                file_text[i+1] = method
        else:
            for i, method in enumerate(self.method_names_alt):
                file_text[i+1] = method

        for metric in self.metrics:
            file_text[0] += ","
            file_text[0] += metric

        for i, method in enumerate(self.method_names):
            dict_mean = {}
            dict_std = {}

            for metric in self.metrics:
                stream_mean = []
                stream_std = []
                for j, stream in enumerate(self.stream_names):

                    temp_data = data[(method, stream)][metric].values
                    mean = np.mean(temp_data)
                    std = np.std(temp_data)

                    stream_mean.append(np.mean(temp_data))
                    stream_std.append(np.std(temp_data))
                    dict_mean[method] = stream_mean
                    dict_std[method] = stream_std

                mean = np.mean(dict_mean[method])
                std = np.std(dict_std[method])
                file_text[i+1] += r',%0.2f$\pm$%0.2f' % (mean*100,std*100)
            if not filename:
                filename = "sum"
            file_object = open("results/overal_score/%s/%s.csv" % (self.date_time, filename), "w")
            for line in file_text:
                file_object.write(line)
                file_object.write("\n")
            file_object.close()

        print(file_text)
