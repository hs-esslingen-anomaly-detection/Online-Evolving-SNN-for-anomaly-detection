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
            self.path = os.path.join(os.getcwd(), os.path.join('Dataset_003'))
        else:
            self.path = os.path.join(os.getcwd(), os.path.join('data', 'Dataset_003'))
        
        datasetName = "set_003"
        
        normal = list()
        normal.append('normal_art_daily_small_noise') #0
        normal.append('normal_art_daily_sin_noise') #1
        
        
        anomaly = list()
        anomaly.append('anomaly_art_daily_drift') #0
        anomaly.append('anomaly_art_daily_jumpsdown') #1
        anomaly.append('anomaly_art_daily_jumpsup') #2
        anomaly.append('anomaly_art_daily_nojump') #3
        anomaly.append('anomaly_art_daily_peaks') #4
        anomaly.append('anomaly_art_daily_sequence_change') #5
        anomaly.append('anomaly_art_daily_increase_noise') #6
        anomaly.append('anomaly_art_daily_amp_rises') #7
        
        # has other scaling need seperate training!
        anomaly.append('anomaly_art_daily_flatmiddle') #8
        
        
        self.benchmarks = list()

        if dim == 1:
            ## RECHTECK ##
            bench = Benchmark(datasetName + "_1d_b001", normal[0])
            bench.addTestSet(anomaly[0])
            bench.addTestSet(anomaly[1])
            bench.addTestSet(anomaly[2])
            bench.addTestSet(anomaly[3])
            bench.addTestSet(anomaly[4])
            bench.addTestSet(anomaly[5])
            bench.addTestSet(anomaly[6])
            bench.addTestSet(anomaly[7])

            self.benchmarks.append(bench)

            # has other scaling need seperate training!
            self.benchmarks.append(Benchmark(datasetName + "_1d_b002", anomaly[8]).addTestSet(anomaly[8]))

        elif dim == 2:
            ## RECHTECK ##
            bench = Benchmark(datasetName + "_2d_b001", (normal[0], anomaly[0])) #training set
            bench.addTestSet((normal[0],anomaly[0]))
            bench.addTestSet((normal[0],anomaly[1]))
            bench.addTestSet((normal[0],anomaly[2]))
            bench.addTestSet((normal[0],anomaly[3]))
            bench.addTestSet((normal[0],anomaly[4]))
            bench.addTestSet((normal[0],anomaly[5]))
            bench.addTestSet((normal[0],anomaly[6]))
            bench.addTestSet((normal[0],anomaly[7]))
            self.benchmarks.append(bench)

            # has other scaling need seperate training!
            self.benchmarks.append(Benchmark(datasetName + "_2d_b002", (normal[0],anomaly[8])).addTestSet((normal[0],anomaly[8])))
            
            ## SINUS ##
            bench = Benchmark(datasetName + "_2d_b003", (normal[1], anomaly[0]))
            bench.addTestSet((normal[1],anomaly[0]))
            bench.addTestSet((normal[1],anomaly[1]))
            bench.addTestSet((normal[1],anomaly[2]))
            bench.addTestSet((normal[1],anomaly[3]))
            bench.addTestSet((normal[1],anomaly[4]))
            bench.addTestSet((normal[1],anomaly[5]))
            bench.addTestSet((normal[1],anomaly[6]))
            bench.addTestSet((normal[1],anomaly[7]))
            self.benchmarks.append(bench)

            # has other scaling need seperate training!
            self.benchmarks.append(Benchmark(datasetName + "_2d_b004", (normal[1],anomaly[8])).addTestSet((normal[1],anomaly[8])))
    
    
        elif dim == 3:
            ## RECHTECK ##
            bench = Benchmark(datasetName + "_3d_b001", (normal[0], normal[0], anomaly[0])) #training set
            bench.addTestSet((normal[0],normal[0],anomaly[0]))
            bench.addTestSet((normal[0],normal[0],anomaly[1]))
            bench.addTestSet((normal[0],normal[0],anomaly[2]))
            bench.addTestSet((normal[0],normal[0],anomaly[3]))
            bench.addTestSet((normal[0],normal[0],anomaly[4]))
            bench.addTestSet((normal[0],normal[0],anomaly[5]))
            bench.addTestSet((normal[0],normal[0],anomaly[6]))
            bench.addTestSet((normal[0],normal[0],anomaly[7]))
            self.benchmarks.append(bench)

            # has other scaling need seperate training!
            self.benchmarks.append(Benchmark(datasetName + "_3d_b002", (normal[0],normal[0],anomaly[8])).addTestSet((normal[0],normal[0],anomaly[8])))
            
            ## SINUS ##
            bench = Benchmark(datasetName + "_3d_b003", (normal[1], normal[1], anomaly[0])) #training set
            bench.addTestSet((normal[1],normal[1],anomaly[0]))
            bench.addTestSet((normal[1],normal[1],anomaly[1]))
            bench.addTestSet((normal[1],normal[1],anomaly[2]))
            bench.addTestSet((normal[1],normal[1],anomaly[3]))
            bench.addTestSet((normal[1],normal[1],anomaly[4]))
            bench.addTestSet((normal[1],normal[1],anomaly[5]))
            bench.addTestSet((normal[1],normal[1],anomaly[6]))
            bench.addTestSet((normal[1],normal[1],anomaly[7]))
            self.benchmarks.append(bench)

            # has other scaling need seperate training!
            self.benchmarks.append(Benchmark(datasetName + "_3d_b004", (normal[1],normal[1],anomaly[8])).addTestSet((normal[1],normal[1],anomaly[8])))
        
        
        elif dim == 4:
            ## RECHTECK ##
            bench = Benchmark(datasetName + "_4d_b001", (normal[0], normal[0], normal[0], anomaly[0])) #training set
            bench.addTestSet((normal[0],normal[0],normal[0],anomaly[0]))
            bench.addTestSet((normal[0],normal[0],normal[0],anomaly[1]))
            bench.addTestSet((normal[0],normal[0],normal[0],anomaly[2]))
            bench.addTestSet((normal[0],normal[0],normal[0],anomaly[3]))
            bench.addTestSet((normal[0],normal[0],normal[0],anomaly[4]))
            bench.addTestSet((normal[0],normal[0],normal[0],anomaly[5]))
            bench.addTestSet((normal[0],normal[0],normal[0],anomaly[6]))
            bench.addTestSet((normal[0],normal[0],normal[0],anomaly[7]))
            self.benchmarks.append(bench)

            # has other scaling need seperate training!
            self.benchmarks.append(Benchmark(datasetName + "_4d_b002", (normal[0],normal[0],normal[0],anomaly[8])).addTestSet((normal[0],normal[0],normal[0],anomaly[8])))
        
            ## SINUS ##
            bench = Benchmark(datasetName + "_4d_b003", (normal[1], normal[1], normal[1], anomaly[0]))
            bench.addTestSet((normal[1],normal[1],normal[1],anomaly[0]))
            bench.addTestSet((normal[1],normal[1],normal[1],anomaly[1]))
            bench.addTestSet((normal[1],normal[1],normal[1],anomaly[2]))
            bench.addTestSet((normal[1],normal[1],normal[1],anomaly[3]))
            bench.addTestSet((normal[1],normal[1],normal[1],anomaly[4]))
            bench.addTestSet((normal[1],normal[1],normal[1],anomaly[5]))
            bench.addTestSet((normal[1],normal[1],normal[1],anomaly[6]))
            bench.addTestSet((normal[1],normal[1],normal[1],anomaly[7]))
            self.benchmarks.append(bench)

            # has other scaling need seperate training!
            self.benchmarks.append(Benchmark(datasetName + "_4d_b004", (normal[1],normal[1],normal[1],anomaly[8])).addTestSet((normal[1],normal[1],normal[1],anomaly[8])))
        
        else:
            print( "[ERROR] Benchmark with dimension " + str(dim) + " not programmed" )
            exit()


        self.pos = 0
        self.pos_max = len(self.benchmarks) #anzahl der benchmarks
    
