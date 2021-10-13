#!/usr/bin/python3

import numpy as np
import pandas as pd
import random
import os


class SignalUtils:

    def CreatePostTrainSet(trainSet, static=False, modifyAllDimensions=False, staticNoiseStartPos=3/5, signal_period=None):
        """
        Creates a signal with random anomalies to test your model or to train it more effectively.
        - set static to true to compare results (Search Hyper-Parameter)
        - set static to false to train your model with a random noise anomaly

        with signal_period you can set the max noise window (1.75 * period)
        modifyAllDimensions specifies if this functions add noise to one random dimension or to all dimensions
        """
        random.seed(42)
        if trainSet.shape[0] > trainSet.shape[1]:
            modifiedTrainSet = np.copy(trainSet.T)
        else:
            modifiedTrainSet = np.copy(trainSet)

        if staticNoiseStartPos > 23/25:
            staticNoiseStartPos = 23/25
        elif staticNoiseStartPos < 0:
            staticNoiseStartPos = 0

        signalDim = modifiedTrainSet.shape[0]
        trainSize = modifiedTrainSet.shape[1]

        noisePos = random.randint(0, signalDim-1) if not static else (signalDim-1)

        noiseMin = np.min(modifiedTrainSet[noisePos,:])
        noiseMax = np.max(modifiedTrainSet[noisePos,:])
        noiseDif = noiseMax - noiseMin
        noiseMin -= (noiseDif / 10)
        noiseMax += (noiseDif / 10)
        noiseDif = noiseMax - noiseMin

        maxNoiseWindow = int(round(trainSize/8, 0))

        if isinstance(signal_period, (int)):
            if signal_period < trainSize/6:
                maxNoiseWindow = int(1.75 * signal_period)

        noiseWindow = random.randint(1, maxNoiseWindow) if not static else maxNoiseWindow
        startNoise = random.randint(int(round(trainSize * 2/5, 0)), (trainSize - noiseWindow)) if not static else int(round(trainSize * staticNoiseStartPos, 0))
        startOffset = random.randint(0, int(round(trainSize * 1/25, 0))) if not static else 0

        if modifyAllDimensions:
            for i in range(startNoise,startNoise + noiseWindow):
                for j in range(signalDim):
                    modifiedTrainSet[j,i] = random.random() * noiseDif + noiseMin
        else:
            for i in range(startNoise,startNoise + noiseWindow):
                modifiedTrainSet[noisePos,i] = random.random() * noiseDif + noiseMin

        return modifiedTrainSet[:,startOffset:], startOffset

