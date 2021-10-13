
from spirit_module import spirit
from data.Dataset_008 import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import matplotlib.pyplot as plt
import numpy as np

import json

scoreWindowSize = 60

result = {}
result['hyperparameters'] = {'scoreWindowSize': scoreWindowSize}
for dim in [1,2,3,4]:
    result[str(dim)] = {}
    DB = Dataset(dim)
    _, testSets = DB.assembleNextBenchmark()

    r, p, f1 = [], [], []
    for testSet in testSets:
        if dim > 1:
            model = spirit(int(dim), int(scoreWindowSize), int(1), float(0.96), float(0.95), float(0.98), True)
            score = model.run(testSet['value'])
            rec = np.array(model.get_reconstruction())
            anomaly = (np.array(score)>=0.67).astype(int)
            gt = [1.0 if any(x > 0.0 for x in testSet['label'][i,:]) else 0.0 for i in range(testSet['label'].shape[0])]
            p.append( precision_score(gt, anomaly, zero_division=0) )
            r.append( recall_score(gt, anomaly, zero_division=0) )
            f1.append( f1_score(gt, anomaly, zero_division=0) )
        else:
            p.append(-1.0)
            r.append(-1.0)
            f1.append(-1.0)

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
