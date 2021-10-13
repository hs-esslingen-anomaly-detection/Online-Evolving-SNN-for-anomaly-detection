#!/bin/python3

from data.Dataset_008 import Dataset
from OeSNNPythonWrapper import PyOeSNN
from sklearn.metrics import mean_squared_error
from optuna.samplers import TPESampler, CmaEsSampler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import sys
import json
import optuna

study = optuna.load_study(study_name="distributed", storage="mysql://root:Geheim@127.0.0.1/optuna")
trial = study.best_trial

result = {}
result['hyperparameters'] = trial.params
for dim in [1,2,3,4]: # eval dim 1 to 4
    result[str(dim)] = {}
    DB = Dataset(dim)
    _, testSets = DB.assembleNextBenchmark()

    r, p, f1 = [], [], []
    for testSet in testSets:
        SNN = PyOeSNN(
                dim,
                Wsize=trial.params['Wsize'],
                NOsize=trial.params['NOsize'],
                NIsize=trial.params['NIsize'],
                tau=trial.params['tau'],
                weightBias=trial.params['weightBias'],
                sim=trial.params['sim'],
                C=trial.params['C'],
                errorCorrection=trial.params['errorCorrection'],
                anomalyThreshold=trial.params['anomalyThreshold'],
                scoreWindowSize=trial.params['scoreWindowSize'],
                random=False,
                debug=True)

        rec = testSet['value'].copy()
        score = np.zeros(testSet['value'].shape[0])
        for i in range(testSet['value'].shape[0]):
            rec[i,:] = SNN.Predict(testSet['value'][i,:])
            score[i] = SNN.GetClassification()

        anomaly = (score>=trial.params['anomalyThreshold']).astype(int)
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
