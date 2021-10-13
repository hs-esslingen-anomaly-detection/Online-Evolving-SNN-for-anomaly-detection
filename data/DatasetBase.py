#!/usr/bin/python3

import numpy as np
import pandas as pd
import itertools as itert
import os


class Benchmark:

    def __init__(self, name, trainingCombination):
        self.name = name
        self.train = trainingCombination
        self.test = list()

    def setTestSet(self, testCombinationList):
        self.test = testCombinationList

    def addTestSet(self, testCombination):
        self.test.append(testCombination)
        return self


class DatasetBase:

    def getNumberOfBenchmarks(self):
        """
        With this function you can retrieve the number of benchmarks. This is useful for a for loop in your program code.
        """
        return self.pos_max


    def loadValues(self, file, postfix='-test.csv'):
        return np.array(pd.read_csv( os.path.join(self.path, (file + postfix)), index_col=0, parse_dates=True )['value'])


    def loadLabels(self, file):
        return np.array(pd.read_csv( os.path.join(self.path, (file + "-test.csv")), index_col=0, parse_dates=True )['outlier'])


    def getRealOutlier(self, files):
        return np.array( [self.loadLabels(f) for f in files] ).T


    def assembleNextBenchmark(self):
        """
        Call this function to get the next benchmark
        """
        self.pos += 1
        if self.pos >= self.pos_max:
            self.pos = 0

        if self.dim == 1:
            trainSet = self.loadValues(self.benchmarks[self.pos].train, '-train.csv').reshape(-1,1)
            testSets = list()
            for i in range(len(self.benchmarks[self.pos].test)):
                testSets.append({
                    "value": self.loadValues(self.benchmarks[self.pos].test[i]).reshape(-1,1),
                    "label": self.loadLabels(self.benchmarks[self.pos].test[i]).reshape(-1,1),
                    "name": np.array( [self.benchmarks[self.pos].test[i]] ),
                    "id": self.benchmarks[self.pos].name + "_test_" + str(i) + '_' + self.benchmarks[self.pos].test[i]
                    })

            return trainSet, testSets

        else:
            trainSet = np.array([self.loadValues(self.benchmarks[self.pos].train[i], '-train.csv') for i in range(self.dim)]).T
            testSets = list()
            for i in range(len(self.benchmarks[self.pos].test)):
                testSets.append({
                    "value": np.array( [self.loadValues(self.benchmarks[self.pos].test[i][j]) for j in range(self.dim)] ).T,
                    "label": np.array( [self.loadLabels(self.benchmarks[self.pos].test[i][j]) for j in range(self.dim)] ).T,
                    "name": np.array( [self.benchmarks[self.pos].test[i][j] for j in range(self.dim)] ),
                    "id": self.benchmarks[self.pos].name + "_test_" + str(i) + '_' + self.benchmarks[self.pos].test[i][-1]
                    })

            return trainSet, testSets
