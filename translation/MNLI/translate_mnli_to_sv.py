import itertools
import json
import jsonlines
import os
import pandas as pd
import torch
import sys

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from torch.utils.data import DataLoader

DEVICE = 1 if torch.cuda.is_available() else -1

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-sv")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-sv")
translation = pipeline("translation_en_to_se", model=model, tokenizer=tokenizer, device=DEVICE)

def grouper(n, iterable, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)

def translate(name, batch_size=32):

    # Read data into pandas DataFrame.
    if name in ('train',
                'validation_matched',
                'validation_mismatched',
                'test_matched',
                'test_mismatched'
    ):
        df = load_dataset("glue", 'mnli')[name].to_pandas()
    elif name.endswith('.csv') or name.endswith('.tsv'):
        sep = '\t' if fn.endswith('.tsv') else ','
        df = pd.read_csv(fn, sep = sep, on_bad_lines = 'skip', keep_default_na=False, lineterminator='\n')
    elif name.endswith('.jsonl'):
        df = pd.read_json(fn, lines=True, on_bad_lines = 'skip', keep_default_na=False)
    else:
        raise ValueError("Cannot read file format of ", fn)

    # Create a list of sentences to translate from the sentence pairs.
    if 'hypothesis' in df.columns and 'premise' in df.columns:
        sentences = list(itertools.chain.from_iterable([
            (data['premise'], data['hypothesis'])
            for _, data in df[['premise', 'hypothesis']].iterrows()
        ]))
    elif 'sentence1' in df.columns and 'sentence2' in df.columns:
        sentences = list(itertools.chain.from_iterable([
            (data['sentence1'], data['sentence2'])
            for _, data in df[['sentence1', 'sentence2']].iterrows()
        ]))
    else:
        error_msg = """
            Could not find sentence pair columns:
            Either 'sentence1, sentence' or 'hypothesis, premise'
        """
        raise ValueError(error_msg)

    loader = DataLoader(sentences, batch_size)
    translated_sentences = []
    counter = 0
    for b in loader:
        counter += batch_size
        translated = [output['translation_text'] for output in translation(b)]
        translated_sentences += translated
        print("\r", "Translating ", fn, " ...", counter,  "/", len(sentences), end="", flush=True)
    pairs = grouper(2, translated_sentences)
    premises, hypotheses = [], []
    for pair, sample in zip(pairs, sentences):
        p, h = pair
        premises.append(p)
        hypotheses.append(h)
    df['premise'], df['hypothesis'] = premises, hypotheses
    path, filename = os.path.split(fn)
    filename = os.path.splitext(filename)[0]
    newfilename = '%s_sv.tsv' % filename
    newpath = os.path.join(path, newfilename)
    df.to_csv(newpath, index=False, sep='\t')
    print("")
    print("[DONE]")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError("No files to translate...")
    filenames = sys.argv[1:]
    for fn in filenames:
        df = translate(fn, batch_size=4)
