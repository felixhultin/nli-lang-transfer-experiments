""" Provides helper functions to download, load and evaluate Swediagnostics"""


from datasets import ClassLabel, load_dataset

def load_swediagnostics_dataset():
    dataset = load_dataset('csv', data_files={'test': './data/DIAGNOSTICS/swediagnostics-v1.0.csv'}, split='test')
    dataset = dataset.remove_columns(['Premise', 'Hypothesis'])
    dataset = dataset.rename_columns({cn: cn.lower() for cn in dataset.column_names})
    dataset = dataset.rename_column('premise_se', 'premise')
    dataset = dataset.rename_column('hypothesis_se', 'hypothesis')
    return dataset

def load_glue_diagnostics_dataset():
    dataset = load_dataset('csv', data_files={'test': './data/DIAGNOSTICS/diagnostic-full.tsv'}, delimiter='\t', split='test')
    dataset = dataset.rename_columns({cn: cn.lower() for cn in dataset.column_names})
    return dataset

if __name__ == '__main__':
    swediagnostics = load_swediagnostics_dataset()
    glue_diagnostics = load_glue_diagnostics_dataset()
