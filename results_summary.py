import argparse
import json
import itertools
import os.path
import pandas as pd

from typing import List

from sklearn.metrics import accuracy_score, matthews_corrcoef

from load_diagnostics import load_glue_diagnostics_dataset, load_swediagnostics_dataset

LINGUISTICS_CATEGORIES = {
    'Lexical Semantics': [
        'Factivity',
         'Lexical entailment',
         'Morphological negation',
         'Named entities',
         'Quantifiers',
         'Redundancy',
         'Symmetry/Collectivity'
    ],
    'Predicate-Argument Structure': [
        'Active/Passive',
        'Anaphora/Coreference',
        'Coordination scope',
        'Core args',
        'Datives',
        'Ellipsis/Implicits',
        'Genitives/Partitives',
        'Intersectivity',
        'Nominalization',
        'Prepositional phrases',
        'Relative clauses',
        'Restrictivity'
    ],
    'Logic': [
        'Conditionals',
        'Conjunction',
        'Disjunction',
        'Double negation',
        'Downward monotone',
        'Existential',
        'Intervals/Numbers',
        'Negation',
        'Non-monotone',
        'Temporal',
        'Universal',
        'Upward monotone'
    ],
    'Knowledge': [
        'Common sense',
        'World knowledge'
    ],
    'Domain': [
        'ACL', 'Artificial', 'News', 'Reddit', 'Wikipedia'
    ]
}

LINGUISTICS_CATEGORIES_LOWER = [l.lower() for l in LINGUISTICS_CATEGORIES]
FINEGRAINED_CATEGORIES = list(itertools.chain.from_iterable(LINGUISTICS_CATEGORIES.values()))
FINEGRAINED_COARSEGRAINED = {fg: coarsegrained for coarsegrained, finegrained in LINGUISTICS_CATEGORIES.items() for fg in finegrained}

def process_df_backup(diagnostics_df):
    data = []
    for _, row in diagnostics_df.iterrows():
        d = { c:v for c,v in row.iteritems() if c not in LINGUISTICS_CATEGORIES_LOWER}
        finegrained = row[LINGUISTICS_CATEGORIES_LOWER]
        for c in finegrained:
            if isinstance(c, str):
                for fg in c.split(';'):
                    d[fg] = True
        data.append(d)
    return pd.DataFrame(data)

def process_df(diagnostics_df):
    data = []
    for _, row in diagnostics_df.iterrows():
        d = { c:v for c,v in row.iteritems() if c not in LINGUISTICS_CATEGORIES_LOWER}
        finegrained = row[LINGUISTICS_CATEGORIES_LOWER]
        for c in finegrained:
            if isinstance(c, str):
                for fg in c.split(';'):
                    data.append({**d, **{'finegrained': fg}})
        data.append(d)
    return pd.DataFrame(data)

def calculate_metrics(df, task_name):
    rows = []
    for c in FINEGRAINED_CATEGORIES:
        c_df = df[df['finegrained'] == c]
        row = {
            'task': task_name,
            'coarsegrained': FINEGRAINED_COARSEGRAINED[c],
            'finegrained': c,
            'size': len(c_df),
            **c_df['label'].value_counts().to_dict(),
            'acc': accuracy_score(c_df['label'], c_df['prediction']),
            'mcc': matthews_corrcoef(c_df['label'], c_df['prediction']),
        }
        rows.append(row)
    return rows

def get_diagnostics_results(df, task_name):
    return calculate_metrics(process_df(df), task_name)

def get_metrics_or_empty_dict(fp):
    if os.path.isfile(fp):
        with open(fp) as f:
            metrics = json.load(f)
            if 'eval_accuracy' in metrics:
                return metrics 
    return {}

def get_results(experiment_names, tasks):
    dfs = {}
    for e_n in experiment_names:
        if not e_n.endswith('/'):
            e_n += '/'
        data = []
        for t in tasks:
            if "multi" in e_n and t in ("snli", "snli-hard", "mnli-matched", "mnli-mismatched"):
                predict_results_fp = f'{e_n}predict-{t}_sv_results.json'
                eval_results_fp = f'{e_n}eval-{t}_sv_results.json'
            else:
                predict_results_fp = f'{e_n}predict-{t}_results.json'
                eval_results_fp = f'{e_n}eval-{t}_results.json'
            predict_metrics = get_metrics_or_empty_dict(predict_results_fp)
            eval_metrics = get_metrics_or_empty_dict(eval_results_fp)
            metrics = predict_metrics if predict_metrics else eval_metrics
            if not metrics:
                print("Skipping ", t, ". Result files do not exist.")
                continue
            source = 'predict' if predict_metrics else 'eval'
            row = {
                    **{'task': t, 'source': source}, 
                    **{'acc': metrics['eval_accuracy'], 'size': metrics['eval_samples']}
            }
            data.append(row)
            if 'diagnostics' in t:
                predictions_fp = f'{e_n}predictions-{t}.tsv'
                if t == 'swediagnostics':
                    full_df = load_swediagnostics_dataset().to_pandas()
                else:
                    full_df = load_glue_diagnostics_dataset().to_pandas()
                test_preds = pd.read_csv(predictions_fp, delimiter='\t')
                complete_df = full_df.merge(test_preds, right_on = ['index'], left_index=True, how='left')
                complete_df['label'] = complete_df['label_x']
                complete_df = complete_df.drop(columns=['label_x', 'label_y'])
                mcc = matthews_corrcoef(complete_df['label'], complete_df['prediction'])
                row['acc'] = mcc #str(row['acc']) + ' (acc) / ' + str(mcc) + ' (mcc)'
                data += get_diagnostics_results(complete_df, t)
        if 'swediagnostics' in tasks or 'glue_diagnostics' in tasks:
            dfs[e_n] = pd.DataFrame(data).fillna('').set_index(['task', 'coarsegrained', 'finegrained'])
        else:
            dfs[e_n] = pd.DataFrame(data).fillna('').set_index('task')
    return dfs

def write2excel(results_dfs : List[pd.DataFrame]):
    with pd.ExcelWriter('results.xlsx') as writer:
        for path, df in results_dfs.items():
            experiment_name = path.strip('/').split('/')[-1]
            print(experiment_name)
            df.to_excel(writer, sheet_name = experiment_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dirs', type=str, nargs='+',
                        help='Model output dirs to analyze.')
    parser.add_argument('--tasks', type=str, nargs='+',
                        default=['mnli-matched', 'mnli-mismatched', 'snli', 'snli-hard', 'glue_diagnostics', 'swediagnostics'],
                        help='Tasks to analyze.')
    args = parser.parse_args()
    output_dirs, tasks = args.output_dirs, args.tasks
    results = get_results(output_dirs, tasks)
    write2excel(results)