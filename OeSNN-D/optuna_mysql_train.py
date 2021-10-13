#!/bin/python3
# Command line Args: NUM_TRIALS_PER_CORE

from data.Dataset_008 import Dataset
from OeSNNPythonWrapper import PyOeSNN
from sklearn.metrics import mean_squared_error
from optuna.samplers import TPESampler, CmaEsSampler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import sys
import optuna
import gc
from lib.SignalUtils import SignalUtils
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

num_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 2

dim = 2 #optimize only for 2 dimensions
DB = Dataset(dim)
trainSet, testSets = DB.assembleNextBenchmark()

trainSignal, _ = SignalUtils.CreatePostTrainSet(trainSet, static=True, modifyAllDimensions=False)
trainSignal = trainSignal.T

def objective(trial):
    maxWSize = 400
    Wsize= trial.suggest_int('Wsize', 200, maxWSize, 50)
    NOsize= trial.suggest_int('NOsize', 2, 128)
    NIsize= trial.suggest_int('NIsize', 2, 128)
    tau = trial.suggest_uniform('tau', 0.5, 3.0)
    weightBias = trial.suggest_categorical('weightBias', [0.0, 0.05, 0.1])
    sim = trial.suggest_loguniform('sim', 0.005, 0.3)
    C = trial.suggest_loguniform('C', 0.01, 0.3)
    errorCorrection= trial.suggest_loguniform('errorCorrection', 0.3, 0.99)
    anomalyThreshold = trial.suggest_uniform('anomalyThreshold', 0.45, 0.85)
    scoreWindowSize = trial.suggest_int('scoreWindowSize', 1, 120)

    SNN = PyOeSNN(
            dim,
            Wsize=Wsize,
            NOsize=NOsize,
            NIsize=NIsize,
            tau=tau,
            weightBias=weightBias,
            sim=sim,
            C=C,
            errorCorrection=errorCorrection,
            anomalyThreshold=anomalyThreshold,
            scoreWindowSize=scoreWindowSize,
            random=False,
            debug=True)

    mse = mean_squared_error(SNN.PredictAll(trainSignal)[Wsize:], trainSet[Wsize:])
    if SNN.GetUnfiredTrials() > 10 \
        or SNN.GetNumberOutputRepoNeurons() < 2 \
        or np.amax(np.array(SNN.GetFiredLog())) > trainSet.shape[0]/4: \
        return sum(trainSet[:,0])**2 + SNN.GetUnfiredTrials()**2 + np.amax(np.array(SNN.GetFiredLog()))**2
    else: return mse

study = optuna.load_study(study_name="distributed", storage="mysql://root:Geheim@127.0.0.1/optuna")
study.optimize(objective, n_trials=num_trials)
