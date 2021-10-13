#!/bin/env python3
# TODO: Use python pkg to wirte markdown

import json
import os

with open('result.md', 'w') as result:
    result.write('# Ergebnisse\n')

    if os.path.exists('preview.png'):
        result.write('![](./preview.png){width=99%}')

    for f in os.listdir('./result'):
        model_name = f.replace('.json', '')
        with open(os.path.join('result', f), 'r') as json_file:
            data = json.loads(json_file.read())
        result.write('\n\n## Hyperparameter ' + model_name + '\n\n')
        result.write('| Parameter | Wert |\n')
        result.write('| -- |:--:|\n')
        for k in data['hyperparameters'].keys():
            if k == 'anomalyThreshold': continue
            result.write('| ' + k + ' | ' \
                    + ('{:.3f}'.format(data['hyperparameters'][k]) \
                    if type(data['hyperparameters'][k]) == float \
                    else str(data['hyperparameters'][k])) \
                    + ' |\n')

    result_files = [f for f in os.listdir('./result')]
    result_files.sort()
    with open(os.path.join('result', result_files[0]), 'r') as json_file: first_data = json.loads(json_file.read())

    for dim in ['1','2','3','4']:
        result.write('\n\n## Detailergebnisse ' + dim + (' Dimensionional' if dim == '1' else ' Dimensionen') + '\n\n')
        result.write('| Anomalie | ')
        for f in result_files: result.write(f.replace('.json','') + ' |')
        result.write('\n')
        result.write(str('| -- ' + '|:--:'*len(result_files) + '|\n'))
        for k in first_data[dim].keys():
            result.write('| ' + k.replace('set_008_' + dim + 'd_b001_test_', '').replace('_', ' ') + ' | ')
            for f in result_files:
                with open(os.path.join('result', f), 'r') as json_file:
                    data = json.loads(json_file.read())
                result.write(('{:.3f}'.format(data[dim][k]['F1']) if data[dim][k]['F1'] >= 0 else '--') + ' | ')
            result.write('\n')


    result.write('\n\n## Ergebnisse\n\n')
    result.write('Average for all anomaly types:\n\n')
    result.write('| Dimension | ')
    for f in result_files: result.write(f.replace('.json','') + ' |')
    result.write('\n')
    result.write(str('|:--:'*(len(result_files)+1) + '|\n'))
    for dim in ['1','2','3','4', 'average']:
        result.write('| '+ dim + ' | ')
        for f in result_files:
            with open(os.path.join('result', f), 'r') as json_file:
                data = json.loads(json_file.read())
            if dim != 'average':
                result.write(('{:.3f}'.format(data[dim]['average']['F1']) if data[dim]['average']['F1'] >= 0 else '--')  + ' | ')
            else:
                avg_f1 = [data[d]['average']['F1'] for d in ['1','2','3','4'] if data[d]['average']['F1'] >= 0]
                avg_f1 = sum(avg_f1)/len(avg_f1)
                result.write('{:.3f}'.format(avg_f1) + ' | ')
        result.write('\n')


    result.write('\n\nAverage for Dimension 2-4:\n\n')
    result.write('| Anomalie | ')
    for f in result_files: result.write(f.replace('.json','') + ' |')
    result.write('\n')
    result.write(str('| -- ' + '|:--:'*len(result_files) + '|\n'))
    for k in first_data['2'].keys():
        if k == 'average': continue
        result.write('| ' + k.replace('set_008_2d_b001_test_', '').replace('_', ' ') + ' | ')
        for f in result_files:
            with open(os.path.join('result', f), 'r') as json_file:
                data = json.loads(json_file.read())
            avg_f1 = [data[d][k.replace('_2d_', '_' + d + 'd_')]['F1'] for d in ['2','3','4']]
            avg_f1 = sum(avg_f1)/len(avg_f1)
            result.write('{:.3f}'.format(avg_f1) + ' | ')
        result.write('\n')


