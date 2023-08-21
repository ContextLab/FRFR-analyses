import pandas as pd
import numpy as np
import seaborn as sns
import dill as pickle

import h5py
import quail
import os
import warnings
import string

from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import IncrementalPCA as PCA
import statannot


figdir = os.path.join(os.path.split(os.getcwd())[0], 'paper', 'figures', 'source')
if not os.path.exists(figdir):
    os.makedirs(figdir)

from dataloader import feature_groupings
from analyze import analyze_data, filter, reorder_df, rename_features, mini_filter, sort_by_grouping, trajectorize, get_dists, listnum2group, results, analyses, listgroups


colors = {
    'feature rich': '#c0673c',
    'reduced (early)': '#f15a22',
    'reduced (late)': '#f7996c',
    'reduced': '#fddac5',
    'category': '#524fa1',
    'size': '#aca7d3',
    'length': '#cbdb2a',
    'first letter': '#e8eeae',
    'color': '#00a651',
    'location': '#9bd3ae',
    'random': '#407b8d',
    'stabilize': '#00addc',
    'destabilize': '#90d7ee',
    'init': '#d3d3d3',
    'early': '#bdccd4',
    'late': '#fdb913'
}


def combo_lineplot(x, include_conds='all', include_lists='all', fname=None, xlabel=None, ylabel=None, xlim=None, ylim=None, palette=None):
    x, include_conds, include_lists = filter(x, include_conds, include_lists)

    fig = plt.figure(figsize=(3, 2.5))
    ax = plt.gca()
    
    for c in include_conds:
        if palette is None:
            if c.lower() in colors:
                next_palette = [colors[c.lower()]]
            else:
                next_palette = [colors[i.lower()] for i in include_lists]
        else:
            next_palette = palette
    
        x[c].plot(ax=ax, palette=next_palette, legend=False)
    
    if xlabel is not None:
        plt.xlabel(xlabel)
    
    if ylabel is not None:
        plt.ylabel(ylabel)

    if xlim is not None:
        plt.xlim(xlim)
    
    if ylim is not None:
        plt.ylim(ylim)
    
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if fname is not None:
        fig.savefig(os.path.join(figdir, fname + '.pdf'), bbox_inches='tight')
    
    return fig


def combo_fingerprint_plot(x, include_conds='all', include_lists='all', fname=None, ylim=None, palette=None, xlabel='Dimension', ylabel='Clustering score', figsize=(15, 2.5)):
    def melt_by_list(x, cond='-'):
        x = rename_features(x).reset_index()
        x = x.rename({c: c.capitalize() for c in x.columns}, axis=1)
        x['Condition'] = cond
        x = x.rename({'List': 'Condition', 'Condition': 'List'}, axis=1)

        return x.melt(id_vars=['Subject', 'List', 'Condition'], var_name='Dimension', value_name='Clustering score')

    def melt_fingerprint(x, cond='-'):
        rename = {'pos': 'location', 'first_letter': 'first letter', 'firstLetter': 'first letter', 'wordLength': 'length'}
        x = x.data.rename(rename, axis=1).reset_index()
        x = x.rename({c: c.capitalize() for c in x.columns}, axis=1)
        x['Condition'] = cond

        return x.melt(id_vars=['Subject', 'List', 'Condition'], var_name='Dimension', value_name='Clustering score')

    x, include_conds, include_lists = filter(x, include_conds, include_lists)

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    
    force_order = ['feature rich', 'reduced (early)', 'reduced (late)', 'reduced',
                   'category', 'size', 'length', 'first letter', 'color', 'location',
                   'adaptive', 'random', 'stabilize', 'destabilize']        

    palette = []
    fingerprints = []
    order = []    

    for c in [x for x in force_order if x in include_conds]:
        if c == 'adaptive':
            fingerprints.append(melt_by_list(x[c], cond=c))
            palette.extend([colors[x] for x in force_order if x in include_lists])
            order.extend([x for x in force_order if x in include_lists])
        else:
            fingerprints.append(melt_fingerprint(x[c], cond=c))
            palette.append(colors[c])
            order.append(c)
    
    sns.violinplot(data=reorder_df(pd.concat(fingerprints, axis=0), 'Condition', order),
                   x='Dimension', y='Clustering score',
                   hue='Condition',
                   order=['Category', 'Size', 'Length', 'First letter', 'Color', 'Location'],
                   palette=palette, linewidth=1, inner='quartile', scale='width', cut=0)
    
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    ax = plt.gca()
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    plt.legend([], [], frameon=False)

    if fname is not None:
        fig.savefig(os.path.join(figdir, fname + '.pdf'), bbox_inches='tight')
    return fig


def plot_heatmaps(results, include_conds='all', include_lists='all', contrasts=None, fname=None, vmin=0, vmax=1.0, dvmin=-1.0, dvmax=1.0, fontsize=12, width=2.5, height=2, xlabel='', ylabel=''):
    def heatmap(m, title, vmn, vmx, ax, showx=False, showy=False, show_title=False, yprepend='', **kwargs):
        sns.heatmap(m, vmin=vmn, vmax=vmx, ax=ax, cbar=False, **kwargs)

        if showx:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if showy:
            ax.set_ylabel(yprepend + ylabel, fontsize=fontsize)
        if show_title:
            ax.set_title(title, fontsize=fontsize)
        return plt.gca()
    
    if type(include_lists) is list:
        original_lists = include_lists.copy()
    else:
        original_lists = include_lists
    
    results, include_conds, include_lists = mini_filter(results, include_conds=include_conds, include_lists=include_lists)

    # correct condition/list order
    include_conds = sort_by_grouping(include_conds, feature_groupings)
    if type(original_lists) is list:
        include_lists = [x for x in original_lists if x in include_lists]

    if contrasts is None:
        contrasts = {}
    n_rows = len(include_lists) + len(contrasts.keys())
    n_columns = len(include_conds)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns, sharex=True, sharey=True, squeeze=True, figsize=(width * n_columns, height * n_rows))

    for column, cond in enumerate(include_conds):
        if cond not in results.keys():
            continue
        for row, listtype in enumerate(include_lists):
            if listtype not in results[cond].keys():
                continue
            
            try:
                ax = axes[row, column]
            except IndexError:
                ax = axes[max([row, column])]

            heatmap(results[cond][listtype], cond.capitalize(), vmin, vmax, ax, showx=row == n_rows, showy=column == 0, show_title=row == 0, yprepend=listtype.capitalize() + '\n')
        
        for i, c in enumerate(contrasts.items()):
            k, v = c
            if (k not in results[cond].keys()) or (v not in results[cond].keys()):
                continue
            
            r = len(include_lists) + i
            try:
                ax = axes[r, column]
            except IndexError:
                ax = axes[max([r, column])]
            heatmap(results[cond][k] - results[cond][v], '', dvmin, dvmax, ax, showx=(len(include_lists) + i + 1) == n_rows, showy=column == 0, cmap='vlag', center=0, yprepend=f'{k.capitalize()} $-$ {v.lower()}\n')
    
    plt.tight_layout()

    if fname is not None:
        fig.savefig(os.path.join(figdir, fname + '.pdf'), bbox_inches='tight')
    return fig


def accuracy_by_list(x, xlim=[1, 16], ylim=[0, 1], fname=None):
    _, inds = np.unique(x['Condition'], return_index=True)
    conds = [x['Condition'][i] for i in sorted(inds)]
    
    palette = [colors[c] for c in conds]
    fig = plt.figure(figsize=(5.5, 2.5))

    # convert lists to 1-indexed
    x['List'] += 1
    
    sns.lineplot(data=x, x='List', y='Accuracy', hue='Condition', palette=palette, legend=False)
    plt.xlabel('List number', fontsize=12)
    plt.ylabel('Recall probability', fontsize=12)
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    if fname is not None:
        fig.savefig(os.path.join(figdir, fname + '.pdf'), bbox_inches='tight')
    return fig


def fingerprint_scatterplot_by_category(results, include_conds='all', include_lists='all', x_lists=None, y_lists=None, x='fingerprint', y='accuracy', average=False, fname=None, xlabel=None, ylabel=None, xlim=None, ylim=None):
    if x_lists is None:
        x_lists = include_lists
    if y_lists is None:
        y_lists = include_lists

    fingerprints, conds, x_lists = filter(results[x], include_conds, x_lists)
    y_data, tmp_conds, y_lists = filter(results[y], include_conds, y_lists)

    assert all([c in conds for c in tmp_conds]) and all([c in tmp_conds for c in conds]), 'condition mismatch!'

    combined_list = []

    if xlabel is None:
        xlabel = 'Clustering score'
    if ylabel is None:
        if y == 'accuracy':
            ylabel = 'Recall probability'
        else:
            ylabel = y.capitalize()

    conds = sort_by_grouping(conds, feature_groupings)
    for c in conds:
        clustering = rename_features(fingerprints[c])[c].reset_index().drop('List', axis=1).set_index('Subject')

        if y == 'fingerprint' or y == 'corrected fingerprint':
            next_y = rename_features(y_data[c])[c].reset_index().drop('List', axis=1).set_index('Subject')
        else:
            next_y = y_data[c].data.reset_index().drop('List', axis=1).set_index('Subject')

        if average:
            clustering = pd.DataFrame(clustering.mean(axis=0)).T
            next_y = pd.DataFrame(next_y.mean(axis=0)).T
                
        if y == 'fingerprint' or y == 'corrected fingerprint':
            clustering = clustering.rename({c: xlabel}, axis=1)
            next_y = next_y.rename({c: ylabel}, axis=1)
            df = pd.concat([clustering, next_y], axis=1)
        else:
            df = pd.concat([clustering, next_y], axis=1).rename({0: ylabel, c: xlabel}, axis=1)
        df['Condition'] = c
        combined_list.append(df)
    
    combined = pd.concat(combined_list, axis=0)
    
    if average:
        fig = plt.figure(figsize=(1.1 * 3.5, 3.5))
        sns.regplot(data=combined, y=ylabel, x=xlabel, color='k', marker='.', scatter_kws={'s': 0.1}, line_kws={'linewidth': 4})
        sns.scatterplot(data=combined, y=ylabel, x=xlabel, hue='Condition', palette=[colors[c] for c in conds], legend=False, s=500, edgecolor='k', linewidth=4)

        # get rid of all tick labels except second and last, increase font size
        ax = plt.gca()        
        ax.set_xticks([np.round(ax.get_xticks()[1], decimals=1), np.round(ax.get_xticks()[-1], decimals=1)])
        ax.set_yticks([np.round(ax.get_yticks()[1], decimals=1), np.round(ax.get_yticks()[-1], decimals=1)])
        ax.tick_params(axis='both', which='major', labelsize=30)
        plt.setp(ax.spines.values(), linewidth=3)
        ax.xaxis.set_tick_params(width=3)
        ax.yaxis.set_tick_params(width=3)        
    else:
        sns.lmplot(data=combined, y=ylabel, x=xlabel, hue='Condition', palette=[colors[c] for c in conds], height=3.5, aspect=1.1)

    ax = plt.gca()
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()

    if average:
        linewidth = 3
    else:
        linewidth = 1.5

    if ylim[0] < 0:        
        plt.plot(xlim, [0, 0], 'k--', linewidth=linewidth)
    if xlim[0] < 0:
        plt.plot([0, 0], ylim, 'k--', linewidth=linewidth)
    
    plt.ylim(ylim)
    plt.xlim(xlim)

    if xlabel is not None:
       plt.xlabel(xlabel, fontsize=14)
    else:
       plt.xlabel(ax.get_xlabel(), fontsize=14)
    
    if ylabel is not None:
       plt.ylabel(ylabel, fontsize=14)
    else:
       plt.ylabel(ax.get_ylabel(), fontsize=14)
    
    # display average plots as insets (no x and y labels)
    if average:
        print('xlabel: {xlabel}, ylabel: {ylabel}'.format(xlabel=xlabel, ylabel=ylabel))
        ax.set_xlabel('') 
        ax.set_ylabel('')

    fig = plt.gcf()
    if fname is not None:
        fig.savefig(os.path.join(figdir, fname + '.pdf'), bbox_inches='tight')

    return fig


def plot_features(m1, m2, conds, xlim=None, ylim=None, fname=None):
    fig = plt.figure(figsize=(3, 3))

    for c in conds:
        plt.plot([m1.loc[c][0], m2.loc[c][0]], [m1.loc[c][1], m2.loc[c][1]], '-', color=colors[c], linewidth=2, alpha=0.25)
        plt.plot(m1.loc[c][0], m1.loc[c][1], 'o', color=colors[c], markersize=8)
        plt.plot(m2.loc[c][0], m2.loc[c][1], '*', color=colors[c], markersize=11)
    
    if xlim is not None:
        plt.xlim(xlim)
    
    if ylim is not None:
        plt.ylim(ylim)

    plt.xlabel('Component 1', fontsize=14)
    plt.ylabel('Component 2', fontsize=14)

    if fname is not None:
        fig.savefig(os.path.join(figdir, fname + '.pdf'), bbox_inches='tight')


def plot_trajectories(fingerprints, include_conds='all', fname=None, xlim=None, ylim=None, model=PCA, average=True):
    fingerprints, _, _ = filter(fingerprints, include_conds, list(range(16)))
    trajectories = trajectorize(fingerprints, average=average, model=model)

    if xlim is not None:
        if type(xlim[0]) is not list:
            xlim = [xlim, [0.5, 16.5]]
    if ylim is not None:
        if type(ylim[0]) is not list:
            ylim = [ylim, [0.4, 1.1]]

    nconds = len(trajectories)
    fig, ax = plt.subplots(nrows=2, ncols=nconds, sharex=False, sharey=False, figsize=(4 * nconds, 7.75))
    #fig.tight_layout()


    conds = sort_by_grouping(list(trajectories.keys()), feature_groupings)

    for i, c in enumerate(conds):
        # plot trajectories
        x = trajectories[c].values
        
        ax[0, i].plot(x[:, 0], x[:, 1], '-', color=colors[c], linewidth=2, alpha=0.5)
        ax[0, i].plot(x[:8, 0], x[:8, 1], 'o', color=colors[c], markersize=8)
        ax[0, i].plot(x[8:, 0], x[8:, 1], '*', color=colors[c], markersize=11)

        ax[0, i].set_title(c.capitalize(), fontsize=18)

        ax[0, i].set_xlabel('Component 1', fontsize=14)
        if i == 0:
            ax[0, i].set_ylabel('Component 2', fontsize=14)
        else:
            ax[0, i].set_yticklabels([])

        if xlim is not None:
            ax[0, i].set_xlim(xlim[0])
        if ylim is not None:
            ax[0, i].set_ylim(ylim[0])
        
        # plot distances
        dists = get_dists(fingerprints[c].data).reset_index().melt(id_vars='Subject', var_name='List', value_name='Distance')
        dists['List'] += 1 # convert from 0-indexed to 1-indexed
        sns.barplot(data=dists, x='List', y='Distance', ax=ax[1, i], color=colors[c])
        ax[1, i].plot([7.5, 7.5], [0, 1.2], 'k--', linewidth=2)

        ax[1, i].set_xlabel('List', fontsize=14)
        if i == 0:
            ax[1, i].set_ylabel('Distance from\nlist 1 fingerprint', fontsize=14)
        else:
            ax[1, i].set_yticklabels([])
            ax[1, i].set_ylabel('')

        if xlim is not None:
            ax[1, i].set_xlim(xlim[1])
        if ylim is not None:
            ax[1, i].set_ylim(ylim[1])
        
        ax[1, i].set_xticks(list(range(1, 16, 2)))

    if fname is not None:
        fig.savefig(os.path.join(figdir, fname + '.pdf'), bbox_inches='tight')

    return fig, ax


def plot_accuracy_near_boundaries(x, listgroup, width=3, height=3, xlim=None, ylim=None, fname=None):
    def plot_helper(a, color='#000', ax=None, xlabel='Lag', ylabel='Accuracy', title=None, xlim=None, ylim=None):
        if ax is None:
            ax = plt.gca()
        
        sns.lineplot(a.reset_index().melt(id_vars='Subject', var_name='Lag', value_name='Recall probability'), x='Lag', y='Recall probability', ax=ax, color=color, legend=False)
        
        if xlim is None:
            xlim = ax.get_xlim()

        if ylim is None:
            ylim = ax.get_ylim()

        ax.plot([0, 0], ylim, '--', color='#000', linewidth=1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_title(title, fontsize=16)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        return plt.gcf()

    n_rows = len(x)
    n_columns = len(list(x.values())[0])

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns, sharex=True, sharey=True, squeeze=True, figsize=(width * n_columns, height * n_rows))

    for column, cond in enumerate(x.keys()):
        for row, feature in enumerate(x[cond].keys()):
            if row == 0:
                title = cond.capitalize()
            else:
                title = None

            if column == 0:
                ylabel = f'{feature.capitalize()}\nRecall probability'
            else:
                ylabel = None
            
            if row == n_rows - 1:
                xlabel = 'Lag'
            else:
                xlabel = None

            plot_helper(x[cond][feature][listgroup], ax=axes[row, column], xlabel=xlabel, ylabel=ylabel, title=title, color=colors[cond], xlim=xlim, ylim=ylim)
    
    if fname is not None:
        fig.savefig(os.path.join(figdir, fname + '.pdf'), bbox_inches='tight')
    
    return fig


def plot_boundary_density_maps(conds, bounds, behaviors, listgroups, width=3, height=3, behavioral_column=None, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, cmap='viridis', fname=None):
    def plot_helper(bounds, behavior, listgroups, color='#000', ax=None, behavioral_column=None, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, cmap='viridis'):
        if ax is None:
            ax = plt.gca()
        
        df = pd.DataFrame(bounds.cumsum(axis=1)[15]).rename({15: 'Number of boundaries'}, axis=1).reset_index()
        df['List group'] = df.apply(lambda x: listnum2group(x['Subject'], x['List'], listgroups), axis=1)

        behavior = rename_features(behavior)

        if ylabel is None:
            ylabel = str(behavior.columns.values[0]).capitalize()
        
        if xlabel is None:
            xlabel = 'Number of boundaries'
        
        if behavioral_column in behavior.columns:
            df[ylabel] = behavior[behavioral_column].values
        elif behavior.shape[1] == 1:
            df[ylabel] = behavior.values
        else:
            raise ValueError(f'Behavioral data does not have a column named {ylabel}')
        
        # density plot
        sns.kdeplot(data=df, x='Number of boundaries', y=ylabel, cmap=cmap, fill=True, levels=50, ax=ax)
        c = ax.collections[0].get_facecolor()
        ax.set_facecolor(c.reshape(4))

        # regression line
        sns.regplot(data=df, x='Number of boundaries', y=ylabel, color='w', scatter=False, ax=ax)
        
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if xlim is None:
            ax.set_xlim([-1, 15])
        else:
            ax.set_xlim(xlim)
        
        if ylim is None:
            ax.set_ylim([0, 1])
        else:
            ax.set_ylim(ylim)

        if title is not None:
            ax.set_title(title, fontsize=16)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=14)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=14)
        
        return plt.gcf()

    n_rows = len(conds)
    n_columns = len(conds)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns, sharex=True, sharey=True, squeeze=True, figsize=(width * n_columns, height * n_rows))

    if xlabel is None:
        xlabel = 'Number of boundaries'
    
    if type(behavioral_column) is not list:
        behavioral_column = [behavioral_column] * len(conds)
    
    if type(ylabel) is not list:
        ylabel = [ylabel] * len(conds)

    for column, cond in enumerate(conds):
        for row, feature in enumerate(conds):
            if row == 0:
                title = cond.capitalize()
            else:
                title = None

            if column == 0:
                if ylabel[column] is None:
                    ylabel = str(feature).capitalize()
                else:
                    use_ylabel = f'{str(feature).capitalize()}\n{ylabel[row]}'
            else:
                use_ylabel = ''
            
            if row == n_rows - 1:
                use_xlabel = xlabel
            else:
                use_xlabel = ''

            plot_helper(bounds[cond][feature], behaviors[cond], listgroups[cond], ax=axes[row, column], behavioral_column=behavioral_column[row], xlabel=use_xlabel, ylabel=use_ylabel, title=title, xlim=xlim, ylim=ylim, cmap=cmap)
    
    if fname is not None:
        fig.savefig(os.path.join(figdir, fname + '.pdf'), bbox_inches='tight')
    
    return fig


def barplot_helper(clustering_results, x='Condition', y=None, hue=None, palette=None, ax=None, fname=None, width=8, height=3, ylim=None, ref=None, ylabel=None):
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca()

    sns.barplot(data=clustering_results, x=x, y=y, hue=hue, palette=palette, ax=ax)

    xlim = plt.xlim()
    if ref is not None:
        y_ref = clustering_results.query('Condition == @ref')[y].mean()
        plt.plot(xlim, [y_ref, y_ref], '--', color='black', linewidth=1)
        plt.xlim(xlim)

        pairs = [((ref, 'Early'), (c, 'Early')) for c in clustering_results['Condition'].unique() if c != ref]
        pairs.extend([((ref, 'Late'), (c, 'Late')) for c in clustering_results['Condition'].unique() if c != ref])
        pairs.extend([((c, 'Early'), (c, 'Late')) for c in clustering_results['Condition'].unique()])

        statannot.add_stat_annotation(
            ax,
            data=clustering_results,
            x=x,
            y=y,
            box_pairs=pairs,
            test='t-test_paired',
            text_format='star',
            loc='outside',
        )

    if ylim is not None:
        plt.ylim(ylim)
    
    plt.legend([],[], frameon=False)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    xlabel = ax.get_xlabel()

    if ylabel is None:
        ylabel = ax.get_ylabel()

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    

    if fname is not None:
        fig.savefig(os.path.join(figdir, fname + '.pdf'), bbox_inches='tight')
    
    return fig