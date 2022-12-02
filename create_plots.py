import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def create_diagnostics_barchart(df, name : str = 'Placeholder'):
    df = df.reset_index()
    df = df[\
        (df['task'].str.contains('diagnostics')) &\
        (df['coarsegrained'] != '') \
    ]
    g = sns.catplot(
        x = "acc",
        y = "finegrained",
        data = df,
        kind ="bar",
        hue = "coarsegrained")
    ax = g.facet_axis(0, 0)
    for c in ax.containers:
        ax.bar_label(c, fmt='%.2f')
    plt.savefig('plots/tasks/{name}_diagnostics_barchart.png'.format(name=name))


def create_tasks_barchart(df, name : str = 'Placeholder'):
    df = df.reset_index()
    df = df[(df['coarsegrained'] == '') & (df['finegrained'] == '')]
    g = sns.catplot(
        x = "acc",
        y = "task",
        data = df,
        kind ="bar")
    ax = g.facet_axis(0, 0)
    for c in ax.containers:
        ax.bar_label(c, fmt='%.2f')
    plt.savefig('plots/diagnostics/{name}_tasks_barchart.png'.format(name=name))


def plot_results(results_fn : str):
    xls = pd.ExcelFile(results_fn)
    folder = 'plots/diagnostics'
    for sn in xls.sheet_names:
         df = pd.read_excel(
             io=fn,
             sheet_name=sn,
             engine='openpyxl',
             index_col = [0,1,2],
             na_filter=False
         )
         create_diagnostics_barchart(df, sn)
         create_tasks_barchart(df, sn)


if __name__ == '__main__':
    fn = sys.argv[1] if len(sys.argv) < 1 else 'results.xlsx'
    plot_results(fn)

    #plot_diagnostics(fn)
