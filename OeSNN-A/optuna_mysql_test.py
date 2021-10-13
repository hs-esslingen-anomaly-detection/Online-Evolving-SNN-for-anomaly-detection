#!/bin/python3

from data.Dataset_008 import Dataset
from OeSNNPythonWrapper import PyOeSNN
from sklearn.metrics import mean_squared_error
from optuna.samplers import TPESampler, CmaEsSampler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import json
import sys
import optuna

study = optuna.load_study(study_name="distributed", storage="mysql://root:Geheim@127.0.0.1/optuna")
trial = study.best_trial

result = {}
result['hyperparameters'] = trial.params
for dim in [1,2,3,4]: # for all dimension identical!
    result[str(dim)] = {}
    DB = Dataset(dim)
    _, testSets = DB.assembleNextBenchmark()

    r, p, f1 = [], [], []
    for testSet in testSets:
        SNN = PyOeSNN(
                Wsize=trial.params['Wsize'],
                NOsize=trial.params['NOsize'],
                NIsize=trial.params['NIsize'],
                TS=1,
                sim=trial.params['sim'],
                C=trial.params['C'],
                mod=trial.params['mod'],
                errorCorrection=trial.params['errorCorrection'],
                anomalyFactor=trial.params['anomalyFactor'],
                random=False,
                debug=True)

        rec = np.zeros(testSet['value'].shape[0])
        score = np.zeros(testSet['value'].shape[0])
        for i in range(testSet['value'].shape[0]):
            rec[i] = SNN.Predict(testSet['value'][i,-1].reshape(-1))
            score[i] = SNN.GetClassification()

        anomaly = (score>=0.8).astype(int)
        for i in range(int(1.5*trial.params['Wsize'])): anomaly[i] = 0 # Einlernphase
        gt = [1.0 if any(x > 0.0 for x in testSet['label'][i,:]) else 0.0 for i in range(testSet['label'].shape[0])]

        p.append( precision_score(gt, anomaly, zero_division=0) )
        r.append( recall_score(gt, anomaly, zero_division=0) )
        f1.append( f1_score(gt, anomaly, zero_division=0) )

        result[str(dim)][testSet['id']] = {
                'Precission': p[-1],
                'Reall': r[-1],
                'F1': f1[-1]
                }

    result[str(dim)]['average'] = {
                'Precission': sum(p)/len(p),
                'Reall': sum(r)/len(r),
                'F1': sum(f1)/len(f1)
            }

print(result)
with open('result.json', 'w') as f: json.dump(result, f, indent=4)
