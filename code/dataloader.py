import pandas as pd
import numpy as np

import quail
import requests
import os
import warnings

from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import cpu_count


# set up labels and directories
datadir = os.path.join(os.path.split(os.getcwd())[0], 'data', 'eggs')

if not os.path.exists(datadir):
    os.makedirs(datadir)


urls = {
    'exp1': 'https://www.dropbox.com/s/wbihewq631onsj8/exp1.egg?dl=1',
    'exp2': 'https://www.dropbox.com/s/kliq92lta7mvqcc/exp2.egg?dl=1',
    'exp3': 'https://www.dropbox.com/s/3se8ee2fahf9hvc/exp3.egg?dl=1',
    'exp4': 'https://www.dropbox.com/s/9xd2v4fofk1uqv2/exp4.egg?dl=1',
    'exp5': 'https://www.dropbox.com/s/lox770xxs2d7ypm/exp5.egg?dl=1',
    'exp6': 'https://www.dropbox.com/s/afp5ml563b46s6q/exp6.egg?dl=1',
    'exp7': 'https://www.dropbox.com/s/43nq4egicrxc31p/exp7.egg?dl=1',
    'exp8': 'https://www.dropbox.com/s/j7j2bldr24wybwu/exp8.egg?dl=1',
    'exp10': 'https://www.dropbox.com/s/gp3av93kvlzsgpw/exp10.egg?dl=1',
    'exp11': 'https://www.dropbox.com/s/5227s2pg5o8krda/exp11.egg?dl=1',
    'exp12': 'https://www.dropbox.com/s/leqll4o8ih587fa/exp12.egg?dl=1'
    }


descriptions = {
    'exp1': 'feature rich',
    'exp2': 'category',
    'exp3': 'color',
    'exp4': 'length',
    'exp5': 'first letter',
    'exp6': 'location',
    'exp10': 'size',
    'exp7': 'reduced (early)',
    'exp8': 'reduced (late)',                
    'exp12': 'reduced',
    'exp11': 'adaptive'
    }


feature_groupings = {
    'random': ['feature rich', 'reduced (early)', 'reduced (late)', 'reduced'],
    'semantic': ['category', 'size'],
    'lexicographic': ['length', 'first letter'],
    'visual': ['color', 'location'],
    'adaptive': ['random', 'stabilize', 'destabilize']
    }


def grouping(feature):
    try:
        return [k for k, v in feature_groupings.items() if feature in v][0]
    except:
        return None


# (down)load the data from each experimental condition
def load_egg(fname, url=None):
    fname = os.path.join(datadir, fname)
    if os.path.exists(fname):
        print('.', end='')
        return quail.load_egg(fname)

    print('o', end='')
    r = requests.get(url, allow_redirects=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    
    return load_egg(fname)


def fetch_data():
    print('loading data...', end='')
    data = {descriptions[x]: load_egg(f'{x}.egg', u) for x, u in urls.items()}
    print('done!')

    return data


def get_listgroups(data):
    listgroups = {}
    for k, egg in data.items():
        if 'listgroup' not in egg.meta:
            listgroups[k] = ['Early' if i < 8 else 'Late' for i in range(16)]
        else:
            listgroups[k] = egg.meta['listgroup']
    return listgroups


def sort_by_grouping(vals, groupings):
    sorted_vals = []
    for category, exemplars in groupings.items():
        sorted_vals.extend([x for x in exemplars if x in vals])
    
    missing = [v for v in vals if v not in sorted_vals]
    sorted_vals.extend(missing)
    
    return sorted_vals
