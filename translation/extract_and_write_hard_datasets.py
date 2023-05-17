import pandas as pd

def extract_snli_hard_sv():
    # Read original SNLI test set in JSONL to dataframe.
    snli_org_df = pd.read_json('SNLI/snli_1.0/snli_1.0_test.jsonl', lines=True)
    # Read machine translated test set to dataframe.
    snli_sv_df = pd.read_csv('SNLI/test_sv.tsv', delimiter='\t')
    # Concatenate above mentioned dataframes.
    df_concat = pd.concat([snli_org_df, snli_sv_df], axis=1)
    # Read original SNLI hard subset to dataframe.
    snli_hard_df = pd.read_json('SNLI_HARD/snli_1.0_test_hard.jsonl', lines=True)
    # Select rows from
    mapper = {'sentence1': 'premise_en', 'sentence2': 'hypothesis_en'}
    snli_hard_df = df_concat\
        .set_index('pairID')\
        .loc[snli_hard_df['pairID']] \
        .reset_index() \
        .rename(columns=mapper)
    snli_hard_df[snli_sv_df.columns]\
        .to_json('snli_test_hard_sv.jsonl', orient='records', lines=True, force_ascii=False)
    snli_hard_df[list(mapper.values()) + ['label']]\
        .rename(columns={'premise_en': 'premise', 'hypothesis_en': 'hypothesis'}) \
        .to_json('snli_test_hard_en.jsonl', orient='records', lines=True, force_ascii=False)
    return snli_hard_df

def extract_mnli_hard_sv():
    # Read original MNLI (mis)matched tests into dataframe.
    mnli_org_matched_df = pd.read_json(
        'MNLI/GLUE_MNLI/original/multinli_0.9_test_matched_unlabeled.jsonl', 
        lines=True
    )
    mnli_org_mismatched_df = pd.read_json(
        'MNLI/GLUE_MNLI/original/multinli_0.9_test_mismatched_unlabeled.jsonl', 
        lines=True
    )
    # Read original MNLI (mis)matched hard subset tests into dataframe.
    mnli_org_matched_hard_df = pd.read_json(
        'MNLI_HARD/multinli_0.9_test_matched_unlabeled_hard.jsonl', 
        lines=True
    )
    mnli_org_mismatched_hard_df = pd.read_json(
        'MNLI_HARD/multinli_0.9_test_mismatched_unlabeled_hard.jsonl', 
        lines=True
    )
    # Read machine translated test set to dataframe.
    mnli_matched_sv_df = pd.read_csv('MNLI/test_matched_sv.tsv', delimiter='\t')
    mnli_mismatched_sv_df = pd.read_csv('MNLI/test_matched_sv.tsv', delimiter='\t')
    # Concatenate above mentioned dataframes.
    df_matched_concat = pd.concat([mnli_org_matched_df, mnli_matched_sv_df], axis=1)
    df_mismatched_concat = pd.concat([mnli_org_mismatched_df, mnli_mismatched_sv_df], axis=1)
    # Select rows from
    mnli_matched_hard_sv_df = df_matched_concat\
        .set_index('pairID')\
        .loc[mnli_org_matched_hard_df['pairID']] \
        .reset_index()[mnli_matched_sv_df.columns]
    fn = 'mnli_matched_hard_sv.jsonl'
    print(f"Writing file to {fn}")
    mnli_matched_hard_sv_df.to_json(fn, orient='records', lines=True, force_ascii=False)

    mnli_mismatched_hard_sv_df = df_mismatched_concat\
        .set_index('pairID')\
        .loc[mnli_org_mismatched_hard_df['pairID']] \
        .reset_index()[mnli_mismatched_sv_df.columns]
    fn = 'mnli_mismatched_hard_sv.jsonl'
    print(f"Writing file to {fn}")
    mnli_mismatched_hard_sv_df.to_json(fn, orient='records', lines=True, force_ascii=False)

    return {'matched_hard_sv': mnli_matched_hard_sv_df, 'mismatched_hard_sv': mnli_mismatched_hard_sv_df}


if __name__ == '__main__':
    snli_hard_sv_df = extract_snli_hard_sv()
    mnli_hard_sv_df = extract_mnli_hard_sv()