import pandas as pd
import numpy as np
import dill as pickle
import seaborn as sns

import quail
import requests
import os
import warnings

from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from dataloader import fetch_data, get_listgroups, datadir


def apply(egg, analysis, listgroup=None):
    warnings.simplefilter('ignore')

    if listgroup is None and not (analysis == 'fingerprint'):
        if 'listgroup' not in egg.meta:
            listgroup = ['Early' if i < 8 else 'Late' for i in range(16)]
        else:
            listgroup = egg.meta['listgroup']

    if type(analysis) is str:
        kwargs = {}
    else:
        analysis, kwargs = analysis
    
    return egg.analyze(analysis, listgroup=listgroup, parallel=False, **kwargs)

    

def analyze_data(data, analyses):
    results = {}

    print('basic analyses...')
    for a in tqdm(analyses):
        kwargs = {}
        if a == 'fingerprint':
            kwargs['permute'] = True
            kwargs['n_perms'] = 1000
        results[a] = {x: apply(d, [a, kwargs]) for x, d in data.items()}
    
    print('pnr analyses...')
    pnr_results = {}
    for i in tqdm(range(16)):
       pnr_results[i] = {x: apply(d, ['pnr', {'position': i}]) for x, d in data.items()}

    results['pnr'] = pnr_results

    print('accuracy by list...')
    listgroup = list(range(16))
    results['accuracy by list'] = {k: apply(egg, 'accuracy', listgroup) for k, egg in data.items()}

    return results


results_file = os.path.join(datadir, 'analyzed.pkl')

if os.path.exists(results_file):
    with open(results_file, 'rb') as f:
        results, analyses, listgroups = pickle.load(f)
else:
    data = fetch_data()
    analyses = ['lagcrp', 'spc', 'accuracy', 'fingerprint']
    results = analyze_data(data, analyses)
    listgroups = get_listgroups(data)
    
    with open(results_file, 'wb') as f:
        pickle.dump([results, analyses, listgroups], f)