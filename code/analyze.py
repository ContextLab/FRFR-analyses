import dill as pickle

import os
import warnings

# noinspection PyProtectedMember
from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import cpu_count
from tqdm import tqdm

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

    if analysis == 'fingerprint':
        listgroup = list(range(16))
    
    return egg.analyze(analysis, listgroup=listgroup, parallel=False, **kwargs)


# noinspection PyShadowingNames
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

    if analysis == 'fingerprint':
        listgroup = list(range(16))

    return egg.analyze(analysis, listgroup=listgroup, parallel=False, **kwargs)


def analyze_data(analyses=['fingerprint', 'pfr', 'lagcrp', 'spc', 'accuracy'], data=None, listgroups=None, savefile=None):
    if savefile is not None:
        savefile = os.path.join(datadir, savefile)

        if os.path.exists(savefile):
            with open(savefile, 'rb') as f:
                results, analyses, listgroups = pickle.load(f)
                return results, analyses, listgroups
    
    if data is None:
        data = fetch_data()
    
    if listgroups is None:
        listgroups = get_listgroups(data)

    scratch_dir = os.path.join(datadir, 'scratch')
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)

    results = {}

    def apply_wrapper(args):
        a, x, d, tmpfile, kwargs = args
        tmpfile = f'{x}-{tmpfile}'
        print(f'starting {a} analysis for condition {x}...')

        if os.path.exists(os.path.join(scratch_dir, tmpfile)):
            with open(os.path.join(scratch_dir, tmpfile), 'rb') as f:
                r = pickle.load(f)
        else:
            r = apply(d, [a, kwargs])
            with open(os.path.join(scratch_dir, tmpfile), 'wb') as f:
                pickle.dump(r, f)

        results[a][x] = r
        print(f'finished {a} analysis for condition {x}')

    print('basic analyses...')
    for a in tqdm(analyses):
        kwargs = {}
        results[a] = {}

        if a == 'fingerprint':
            kwargs['permute'] = True
            kwargs['n_perms'] = 500
            tmpfile = f'{a}-{kwargs["permute"]}-{kwargs["n_perms"]}.pkl'
        else:
            if a == 'pfr':
                kwargs['position'] = 0
            tmpfile = f'{a}.pkl'

        # save out temp files
        if a not in ['pfr', 'pnr']:
            with Pool(min([cpu_count(), len(data)])) as p:
                p.map(apply_wrapper, [[a, x, d, tmpfile, kwargs] for x, d in data.items()])
        else:
            [apply_wrapper([a, x, d, tmpfile, kwargs]) for x, d in data.items()]
        
        # load in temp files and update results
        for x in data.keys():
            next_tmpfile = f'{x}-{tmpfile}'
            with open(os.path.join(scratch_dir, next_tmpfile), 'rb') as f:
                results[a][x] = pickle.load(f)

    print('pnr analyses...')
    pnr_results = {}
    for i in tqdm(range(16)):
        pnr_results[i] = {x: apply(d, ['pnr', {'position': i}]) for x, d in data.items()}

    results['pnr'] = pnr_results

    print('accuracy by list...')
    listgroup = list(range(16))
    results['accuracy by list'] = {k: apply(egg, 'accuracy', listgroup) for k, egg in data.items()}

    if savefile is not None:
        with open(savefile, 'wb') as f:
            pickle.dump([results, analyses, listgroups], f)

    return results, analyses, listgroups
