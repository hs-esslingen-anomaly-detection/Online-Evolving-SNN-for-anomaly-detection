#!/usr/bin/python3

import numpy as np
import pandas as pd
import itertools as itert
import os

if os.getcwd().endswith('data'):
    from DatasetBase import DatasetBase, Benchmark
else:
    from data.DatasetBase import DatasetBase, Benchmark


class Dataset(DatasetBase):

    def __init__(self, dim=2):
        self.dim = dim

        if os.getcwd().endswith('data'):
            self.path = os.path.join(os.getcwd(), os.path.join('Dataset_008'))
        else:
            self.path = os.path.join(os.getcwd(), os.path.join('data', 'Dataset_008'))

        datasetName = "set_008"

        normal = list()
        normal.append('sin_no_anomaly')
        normal.append('sin_no_anomaly')
        normal.append('sin_no_anomaly')
        normal.append('sin_no_anomaly')


        anomaly = list()
        anomaly.append('sin_drift') #0
        anomaly.append('sin_lower_amplitude') #1
        anomaly.append('sin_higher_amplitude') #2
        anomaly.append('sin_dropout') #3
        anomaly.append('sin_single_peaks') #4
        anomaly.append('sin_frequency') #5
        anomaly.append('sin_increase_noise') #6
        anomaly.append('sin_amplitude') #7
        anomaly.append('sin_flatmiddle') #8



        self.benchmarks = list()

        if dim == 1:
            bench = Benchmark(datasetName + "_1d_b001", (normal[0]))
            bench.addTestSet((anomaly[0]))
            bench.addTestSet((anomaly[1]))
            bench.addTestSet((anomaly[2]))
            bench.addTestSet((anomaly[3]))
            bench.addTestSet((anomaly[4]))
            bench.addTestSet((anomaly[5]))
            bench.addTestSet((anomaly[6]))
            bench.addTestSet((anomaly[7]))
            bench.addTestSet((anomaly[8]))
            self.benchmarks.append(bench)


        elif dim == 2:
            bench = Benchmark(datasetName + "_2d_b001", (normal[0], normal[0]))
            bench.addTestSet((normal[0],anomaly[0]))
            bench.addTestSet((normal[0],anomaly[1]))
            bench.addTestSet((normal[0],anomaly[2]))
            bench.addTestSet((normal[0],anomaly[3]))
            bench.addTestSet((normal[0],anomaly[4]))
            bench.addTestSet((normal[0],anomaly[5]))
            bench.addTestSet((normal[0],anomaly[6]))
            bench.addTestSet((normal[0],anomaly[7]))
            bench.addTestSet((normal[0],anomaly[8]))
            self.benchmarks.append(bench)


        elif dim == 3:
            bench = Benchmark(datasetName + "_3d_b001", (normal[0], normal[0], normal[0]))
            bench.addTestSet((normal[0],normal[1],anomaly[0]))
            bench.addTestSet((normal[0],normal[1],anomaly[1]))
            bench.addTestSet((normal[0],normal[1],anomaly[2]))
            bench.addTestSet((normal[0],normal[1],anomaly[3]))
            bench.addTestSet((normal[0],normal[1],anomaly[4]))
            bench.addTestSet((normal[0],normal[1],anomaly[5]))
            bench.addTestSet((normal[0],normal[1],anomaly[6]))
            bench.addTestSet((normal[0],normal[1],anomaly[7]))
            bench.addTestSet((normal[0],normal[1],anomaly[8]))
            self.benchmarks.append(bench)

        elif dim == 4:
            bench = Benchmark(datasetName + "_4d_b001", (normal[2], normal[0], normal[1], normal[0]))
            bench.addTestSet((normal[2],normal[0],normal[1],anomaly[0]))
            bench.addTestSet((normal[2],normal[0],normal[1],anomaly[1]))
            bench.addTestSet((normal[2],normal[0],normal[1],anomaly[2]))
            bench.addTestSet((normal[2],normal[0],normal[1],anomaly[3]))
            bench.addTestSet((normal[2],normal[0],normal[1],anomaly[4]))
            bench.addTestSet((normal[2],normal[0],normal[1],anomaly[5]))
            bench.addTestSet((normal[2],normal[0],normal[1],anomaly[6]))
            bench.addTestSet((normal[2],normal[0],normal[1],anomaly[7]))
            bench.addTestSet((normal[2],normal[0],normal[1],anomaly[8]))
            self.benchmarks.append(bench)


        else:
            print( "[ERROR] Benchmark with dimension " + str(dim) + " not programmed" )
            exit()


        self.pos = 0
        self.pos_max = len(self.benchmarks) #anzahl der benchmarks

