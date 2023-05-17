import os
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from sklearn.metrics import accuracy_score, matthews_corrcoef
from typing import List

def format_modelname(modelname : str):
    return {
        # English models
        'bert-base-cased_mnli': 'BERT.mnli',
        'bert-base-cased_snli': 'BERT.snli',
        # Multilingual models
        'bert-base-multilingual-cased_mnli': 'mBERT.mnli',
        'bert-base-multilingual-cased_snli': 'mBERT.snli',
        'bert-base-multilingual-cased_mnli_sv_full': 'mBERT.mnli-sv',
        'bert-base-multilingual-cased_snli_sv': 'mBERT.snli-sv',
        # Swedish models
        'bert-base-swedish-cased_mnli_sv_full': 'KB-BERT.mnli-sv',
        'bert-base-swedish-cased_snli_sv': 'KB-BERT.snli-sv',
    }[modelname]

def create_diagnostics_barchart(df, name : str, lang : str, metric : str ='mcc'):
    task = 'swediagnostics' if lang == 'sv' else 'glue_diagnostics'
    df = df.reset_index()
    df = df[\
        (df['task'] == task) &\
        (df['coarsegrained'] != '')
    ]
    # Temporary hack
    df = df.drop(df[df.source != ''].index)
    df = df[['coarsegrained', 'finegrained', metric]]
    g = sns.barplot(
        x = metric,
        y = "finegrained",
        data = df,
        hue = "coarsegrained",
        errorbar=None,
        dodge = False
    )
    g.legend_.set_title(None)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    g.set(xlabel=None, ylabel=None, xlim=[-1,1])
    for i in g.containers:
        g.bar_label(i, fmt='%.2f')
    plt.tight_layout()
    plt.savefig(f'plots/diagnostics/{name}_{task}_{metric}_hbarplot.svg')
    plt.savefig(f'plots/diagnostics/{name}_{task}_{metric}_hbarplot.png')
    plt.clf()

def read_results(fn: str, model_path : str):
    df = pd.read_excel(
        io=fn,
        sheet_name=model_path,
        engine='openpyxl',
        index_col = [0,1,2],
        na_filter=False
    )
    return df

def filter_diagnostic_results(df: pd.DataFrame, task: str = None):
    df = df.reset_index()
    if task:
        df = df[df['task'] == task]
    df = df[\
        (df['coarsegrained'] != '') &\
        (df['coarsegrained'] != 'Domain')
    ]
    df = df.drop(df[df.source != ''].index)
    df = df[['coarsegrained', 'finegrained', 'mcc']]
    return df

def create_overlapping_diagnostics_barchart(fn : str, model_path_1 : str, model_path_2: str, metric : str ='mcc', show_xticks: bool = True):
    df0 = filter_diagnostic_results(
        read_results(fn, model_path_1), 
        'swediagnostics' if any(s in model_path_1 for s in ('sv', 'multi')) else 'glue_diagnostics'
    )
    df1 = filter_diagnostic_results(
        read_results(fn, model_path_2), 
        'swediagnostics' if any(s in model_path_2 for s in ('sv', 'multi'))  else 'glue_diagnostics'
    )
    #
    #sns.reset_orig()
    #sns.set_style(style=None, rc=None)
    g0 = sns.barplot(
        x = metric,
        y = "finegrained",
        data = df0,
        hue = "coarsegrained",
        errorbar=None,
        dodge = False,
        width=0.8,
        linewidth=20,
        alpha=0.5
    )
   
    g0.set(xlabel=None, ylabel=None, xlim=[-0.7,1])
    g1 = sns.barplot(
        x = metric,
        y = "finegrained",
        data = df1,
        hue = "coarsegrained",
        errorbar= None,
        dodge = False,
        ax=g0,
        width=0.4
    )
    g0.set(ylabel=None, xlabel=metric.upper())
    sns.set(font_scale=0.5)
    labels_df0 = []
    labels_df1 = []
    for i in range(len(df0)):
        df0_i_mcc = df0.iloc[i].mcc
        df1_i_mcc = df1.iloc[i].mcc
        diff = round(abs(df0_i_mcc - df1_i_mcc), 2)
        # When both mcc values are negative.
        if df0_i_mcc < 0 and df1_i_mcc < 0:
            if df0_i_mcc <= df1_i_mcc:
                labels_df0.append(diff)
                labels_df1.append("")
            else:
                labels_df1.append(diff)
                labels_df0.append("")
            continue 
        # When there is at least one positive mcc value.
        if df0_i_mcc >= df1_i_mcc:
            labels_df0.append(diff)
            labels_df1.append("")
        else:
            labels_df1.append(diff)
            labels_df0.append("")
    for c in g0.containers[:4]:
        g0.bar_label(c, labels_df0)
    for c in g1.containers[4:]:
        g1.bar_label(c, labels_df1)
    coarsegrained_leg = plt.legend(labels=df1.coarsegrained.unique(), loc='upper left')
    if not show_xticks:
        import pdb; pdb.set_trace()
        g0.yaxis.set_ticks([], [])
    g0.add_artist(coarsegrained_leg)
    model_name_1, model_name_2 = format_modelname(model_path_1), format_modelname(model_path_2) 
    transparent_line = mpatches.Patch(color='black', alpha=0.5, label=f'{model_name_1} (transparent)')
    thick_line = mpatches.Patch(color='black', alpha=1, label=f'{model_name_2} (solid)')
    model_leg = plt.legend(handles=[transparent_line, thick_line], loc='upper left', bbox_to_anchor=(0,0.85))
    plt.tight_layout()
    plt.rcParams["font.family"] = "Times New Roman"
    #plt.rcParams["font.size"] = 22
    plt.savefig(f'plots/diagnostics/{model_path_1}-COMPARED_TO-{model_path_2}_{metric}_hbarplot_overlapped.svg')
    plt.savefig(f'plots/diagnostics/{model_path_1}-COMPARED_TO-{model_path_2}_{metric}_hbarplot_overlapped.png')
    plt.savefig(f'lololol.svg')
    plt.savefig(f'lololol.png')
    #plt.savefig(f'plots/diagnostics/{model_path_1}-COMPARED_TO-{model_path_2}_{metric}_hbarplot_overlapped.eps', format='eps')
    plt.clf()
    sns.reset_defaults()

def create_tasks_df(results_fn : str):
    xls = pd.ExcelFile(results_fn)
    sn_dfs = []
    for sn in xls.sheet_names:
        sn_df = pd.read_excel(
            io=fn,
            sheet_name=sn,
            engine='openpyxl'
        ).fillna('').set_index('task', 'coarsegrained', 'finegrained')
        sn_df = sn_df.reset_index()
        sn_df = sn_df[(sn_df['coarsegrained'] == '') & (sn_df['finegrained'] == '')]
        sn_df = sn_df[['task', 'acc']]
        sn_df['Model/Train'] = sn
        sn_df['Model/Train'] = sn_df['Model/Train'].map(format_modelname)
        sn_dfs.append(sn_df)
    return pd.concat(sn_dfs).pivot('Model/Train', 'task', 'acc')

def plot_barchart(results_fn : str, to_compare: List[str]):
    df = create_tasks_df(fn)
    to_drop = ['snli-hard']
    df = df.drop(columns=['glue_diagnostics', 'swediagnostics'] + to_drop)
    df = df.reset_index()
    df = df[df['Model/Train'].isin(to_compare)]
    df = df.melt(id_vars=['Model/Train'], var_name='task', value_name='score')
    df = df.rename(columns={'Model/Train': 'Model', 'score': 'ACC'})
    sns.set(font_scale=0.7)
    sns.set_style("whitegrid")
    hue_order = [
        'BERT.mnli',
        'KB-BERT.mnli-sv',
        'mBERT.mnli',
        'mBERT.mnli-sv',
        'BERT.snli',
        'KB-BERT.snli-sv',
        'mBERT.snli',
        'mBERT.snli-sv',
    ]
    g = sns.barplot(df, x = 'task', y = 'ACC', hue='Model', hue_order=hue_order)
    for i in g.containers:
        g.bar_label(i, fmt='%.2f')
    g.set(xlabel=None, ylim=[0.5,1])
    g.legend().set_title(None)
    sns.move_legend(g, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15))
    #sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('plots/tasks/all_barchart.png')
    plt.savefig('plots/tasks/all_barchart.eps', format='eps')
    plt.clf()
    sns.reset_defaults()
    return df

def plot_diagnostics(results_fn : str, to_compare: List[str]):
    df = create_tasks_df(fn)
    df = df[['glue_diagnostics', 'swediagnostics']]
    df = df.rename(columns={'glue_diagnostics': 'GLUE Diagnostics (English)', 'swediagnostics': 'SweDiagnostics (Swedish)'})
    df = df.reset_index()
    df = df[df['Model/Train'].isin(to_compare)]
    df = df.melt(id_vars=['Model/Train'], var_name='task', value_name='score')
    df = df.rename(columns={'Model/Train': 'Model', 'score': 'MCC'})
    sns.set(font_scale=0.75)
    sns.set_style("whitegrid")
    hue_order = [
        'BERT.mnli',
        'mBERT.mnli',
        'mBERT.mnli-sv',
        'KB-BERT.mnli-sv',
        'BERT.snli',
        'mBERT.snli',
        'mBERT.snli-sv',
        'KB-BERT.snli-sv'
    ]
    g = sns.barplot(df, x = 'Model', y='MCC', hue='task', order=hue_order)
    g.tick_params(axis='x', rotation=90)
    g.set(xlabel=None, ylim=[0,0.4])
    g.legend().set_title(None)
    for i in g.containers:
        g.bar_label(i, fmt='%.2f')
    sns.move_legend(g, "upper right")#, bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('plots/tasks/all_diagnostics.eps', format='eps')
    plt.savefig('plots/tasks/all_diagnostics.png')
    plt.clf()
    sns.reset_defaults()
    return df

def plot_average_mean_diagnostics_heatmap(result_folders : List[str] = os.listdir('model_results'), metric : str = 'acc'):
    glue_diagnostics_fn = 'predictions-glue_diagnostics.tsv'
    swediagnostics_fn = 'predictions-swediagnostics.tsv'
    prediction_fns = [
        f'model_results/{dir}/{swediagnostics_fn}'
        if dir != 'bert-base-cased_mnli' else f'model_results/{dir}/{glue_diagnostics_fn}'
        for dir in result_folders
    ]
    prediction_dfs = [pd.read_csv(fp, delimiter='\t') for fp in prediction_fns] + ["Gold"]
    gold_labels = prediction_dfs[0]['label']
    results = []
    for pf1 in prediction_dfs:
        pf1_results = []
        for pf2 in prediction_dfs:
            pf1_labels = gold_labels if type(pf1) != pd.DataFrame else pf1["prediction"]
            pf2_labels = gold_labels if type(pf2) != pd.DataFrame else pf2["prediction"]
            m = accuracy_score(pf1_labels, pf2_labels) if metric == 'acc' else matthews_corrcoef(pf1_labels, pf2_labels)
            pf1_results.append(m)
        results.append(pf1_results)
    model_names = [format_modelname(rf) for rf in result_folders] + ['Gold']
    sns.set(font_scale=1.2)
    g = sns.heatmap(results, annot=True, xticklabels=model_names, yticklabels=model_names)
    g.tick_params(axis='y', rotation=0)
    g.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.savefig(f'plots/tasks/compare_diagnostics_heatmap_{metric}.png')
    plt.savefig(f'plots/tasks/compare_diagnostics_heatmap_{metric}.eps', format='eps')
    plt.clf()

def plot_heatmap(fn: str):
    df = create_tasks_df(fn)
    sns.heatmap(df, annot=True)
    plt.tight_layout()
    plt.savefig('plots/tasks/all_heatmap.png')
    plt.savefig('plots/tasks/all_heatmap.svg')
    plt.clf()
    sns.reset_defaults()
    return df

def create_diagnostic_statistics_table(fn):
    df = pd.read_excel(
             io=fn,
             sheet_name='bert-base-cased_mnli',
             engine='openpyxl',
             index_col = [0,1,2],
             na_filter=False
    ).reset_index()
    df = df[df['task'] == 'glue_diagnostics']
    df = df\
        .set_index(['coarsegrained', 'finegrained']) \
        .drop(columns=['task', 'source', 'acc', 'mcc'])\
        .drop(('', ''))
    df.to_excel('diagnostics_categories_stats.xlsx')
    return df
        
if __name__ == '__main__':
    # Filename of results file.
    fn = sys.argv[1] if len(sys.argv) < 1 else 'results.xlsx'    
    # Plot task results (accuracy)
    to_compare = [
        'BERT.mnli',
        'BERT.snli',
        'mBERT.mnli',
        'mBERT.snli',
        'mBERT.mnli-sv',
        'mBERT.snli-sv',
        'KB-BERT.mnli-sv',
        'KB-BERT.snli-sv'
    ]
    plot_barchart(fn, to_compare)
    # Plot heatmap
    to_compare = [
        'bert-base-swedish-cased_mnli_sv_full',
        'bert-base-multilingual-cased_mnli',
        'bert-base-multilingual-cased_mnli_sv_full',
        'bert-base-cased_mnli'
    ]
    plot_average_mean_diagnostics_heatmap(to_compare, metric = 'acc')
    plot_average_mean_diagnostics_heatmap(to_compare, metric = 'mcc')
    
    to_compare = [
        'BERT.mnli',
        'BERT.snli',
        'mBERT.mnli',
        'mBERT.snli',
        'mBERT.mnli-sv',
        'mBERT.snli-sv',
        'KB-BERT.mnli-sv',
        'KB-BERT.snli-sv'
    ]

    plot_diagnostics(fn, to_compare)

    # Plot diagnostics comparable barchart (matthew correlation coefficient)
    #create_overlapping_diagnostics_barchart(fn, 'bert-base-swedish-cased_mnli_sv_full', 'bert-base-cased_mnli')
    #create_overlapping_diagnostics_barchart(fn, 'bert-base-multilingual-cased_mnli_sv_full', 'bert-base-cased_mnli')
    create_overlapping_diagnostics_barchart(fn, 'bert-base-multilingual-cased_mnli', 'bert-base-cased_mnli', show_xticks=False)
    #create_overlapping_diagnostics_barchart(fn, 'bert-base-multilingual-cased_mnli', 'bert-base-swedish-cased_mnli_sv_full')