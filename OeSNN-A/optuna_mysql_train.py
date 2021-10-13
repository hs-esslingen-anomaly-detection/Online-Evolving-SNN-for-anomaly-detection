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
    sim = trial.suggest_loguniform('sim', 0.005, 0.3)
    C = trial.suggest_loguniform('C', 0.01, 0.3)
    mod = trial.suggest_loguniform('mod', 0.005, 0.99)
    errorCorrection= trial.suggest_loguniform('errorCorrection', 0.3, 0.99)
    anomalyFactor = trial.suggest_int('anomalyFactor', 2, 12)

    SNN = PyOeSNN(
            Wsize=Wsize,
            NOsize=NOsize,
            NIsize=NIsize,
            TS=1,
            sim=sim,
            C=C,
            mod=mod,
            errorCorrection=errorCorrection,
            anomalyFactor=anomalyFactor,
            random=False,
            debug=True)

    rec = np.zeros(trainSignal.shape[0])
    score = np.zeros(trainSignal.shape[0])
    for i in range(trainSignal.shape[0]):
        rec[i] = SNN.Predict(trainSignal[i,-1])
        score[i] = SNN.GetClassification()

    mse = mean_squared_error(rec[Wsize:], trainSet[Wsize:,-1])
    unfired_trials = len([1 for x in np.array(SNN.GetFiredLog()) if x < 0])
    if unfired_trials > 10: return sum(trainSet[:,0])**2 + unfired_trials**2
    else: return mse

study = optuna.load_study(study_name="distributed", storage="mysql://root:Geheim@127.0.0.1/optuna")
study.optimize(objective, n_trials=num_trials)
