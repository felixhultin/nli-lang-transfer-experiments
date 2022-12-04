#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys

import datasets
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from get_args import ModelArguments, DataTrainingArguments
from load_diagnostics import load_swediagnostics_dataset, load_glue_diagnostics_dataset

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.14.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

tasks = "mnli mrpc qnli qqp rte sst stsb wnli boolq cb copa".split()  # copa is special
sglues = tasks[-3:]

task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "snli": ("premise", "hypothesis"),
    "qnli": ("premise", "hypothesis"),
    "rte": ("premise", "hypothesis"),
    "wnli": ("premise", "hypothesis")
}

logger = logging.getLogger(__name__)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, trainisupng_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    tasks = "mnli mrpc qnli qqp rte sst stsb wnli boolq cb copa".split() 
    if data_args.task_name is not None and data_args.task_name in tasks:
        if data_args.task_name in sglues:
            metric = load_metric('super_glue', data_args.task_name)
        else:
            if data_args.task_name == "sst":
                metric = load_metric('glue', "sst2")
            else:
                metric = load_metric('glue', data_args.task_name)
    else:
        metric = load_metric('accuracy')
    
    if data_args.lang == 'sv':
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file}
        if data_args.task_name == 'mnli':
            data_files['validation_matched'] = data_args.validation_matched_file
            data_files['validation_mismatched'] = data_args.validation_mismatched_file
        else:
            data_files['validation'] =  data_args.validation_file
        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if any([data_args.test_file, data_args.test_matched_file, data_args.test_mismatched_file]):
                train_extension = data_args.train_file.split(".")[-1]
                if data_args.task_name == 'mnli':
                    test_matched_extension = data_args.test_matched_file.split(".")[-1]
                    test_mismatched_extension = data_args.test_mismatched_file.split(".")[-1]
                    assert(
                        test_matched_extension == test_mismatched_extension
                    ), "test_matched file and test_mismatched file should have the same extension (csv/tsv or json)"
                    test_extension = test_matched_extension
                else:
                    test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                if data_args.task_name == 'mnli':
                    data_files['test_matched'] = data_args.test_matched_file
                    data_files['test_mismatched'] = data_args.test_mismatched_file
                else:
                    data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")
        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
    else:
        if data_args.task_name in tasks:
            if data_args.task_name in sglues:
                raw_datasets = load_dataset('super_glue', data_args.task_name)
            else:
                raw_datasets = load_dataset('glue', data_args.task_name)
        else:
            raw_datasets = load_dataset(data_args.task_name)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = ['entailment', 'neutral', 'contradiction']
            num_labels = len(label_list)
            #num_labels = 3
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["trainsuper_glue"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        try:
            sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
        except ValueError:
            sentence1_key = task_to_keys[data_args.task_name][0]
            sentence2_key = None
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        #model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key], ) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result
    
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        if data_args.task_name == 'snli':
            train_dataset = train_dataset.filter(lambda example: example['label'] != -1)

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        if data_args.task_name == 'snli':
            eval_dataset = eval_dataset.filter(lambda example: example['label'] != -1)

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            import pdb; pdb.set_trace()
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        if data_args.task_name == 'snli':
            predict_dataset = predict_dataset.filter(lambda example: example['label'] != -1)
    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None and data_args.task_name in tasks:
        # metric = load_metric("glue", data_args.task_name)
        if data_args.task_name in sglues:
            metric = load_metric('super_glue', data_args.task_name)
        else:
            if data_args.task_name == "sst":
                metric = load_metric('glue', "sst2")
            else:
                metric = load_metric('glue', data_args.task_name)
    else:
        metric = load_metric("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
             tasks.append("mnli-mm")
             eval_datasets.append(raw_datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval-" + task , metrics)
            trainer.save_metrics("eval-" + task, metrics)
    
    def load_predict_dataset(test_task : str):
        glue_tasks = "mnli mrpc qnli qqp rte sst stsb wnli ax boolq cb copa".split()
        sglue_tasks = glue_tasks[-3:]
        if test_task.endswith('json'):
            return load_dataset("json", data_files={'test': test_task}, cache_dir=model_args.cache_dir)['test']
        elif test_task.endswith(('csv', 'tsv')):
            delimiter = ',' if test_task.endswith('csv') else '\t'
            return load_dataset("csv", data_files={'test': test_task}, delimiter=delimiter, cache_dir=model_args.cache_dir)['test']
        elif test_task in ('glue_diagnostics', 'swediagnostics'):
            ds = load_glue_diagnostics_dataset() if test_task == 'glue_diagnostics' else load_swediagnostics_dataset()
            def remap_labels(example):
                example['label'] = config.label2id[example['label']]
                return example
            ds = ds.map(remap_labels)
            return ds
        elif test_task in glue_tasks:
            if test_task in sglue_tasks:
                return load_dataset('super_glue', test_task)['test']
            else:
                return load_dataset('glue', test_task)['test']
        else:
            return load_dataset(test_task)['test']

    
    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        #tasks = [data_args.task_name] + data_args.test_tasks
        #predict_datasets = [predict_dataset]
        predict_datasets = []
        for t in data_args.test_tasks:
            if t != data_args.task_name:
                non_task_predict_dataset = load_predict_dataset(t)
                non_task_predict_dataset = non_task_predict_dataset.map(
                    preprocess_function,
                    batched=True,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )
                predict_datasets.append(non_task_predict_dataset)

        if data_args.task_name == "mnli":
             tasks.append("mnli-mm")
             predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, test_task in zip(predict_datasets, data_args.test_tasks):
            labels = predict_dataset["label"] # Save for later.
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            if -1 in predict_dataset["label"]:
                if set(predict_dataset['label']) != {-1}:
                    predict_dataset = predict_dataset.filter(lambda example: example["label"] != -1)
                else:
                    predict_dataset = predict_dataset.remove_columns("label")
                               
            metrics = trainer.evaluate(eval_dataset=predict_dataset)
            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(predict_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(predict_dataset))

            # Hack solution
            # TODO: Fix later
            if 'mnli' in test_task.lower():
                test_task = 'mnli-mismatched' if 'mismatched' in test_task else 'mnli-matched'  
            elif 'snli' in test_task.lower():
                test_task = 'snli'
            else:
                test_task = test_task

            trainer.log_metrics("predict-" + test_task, metrics)
            trainer.save_metrics("predict-" + test_task, metrics)

            # Write predictions to file.
            if test_task in ('swediagnostics', 'glue_diagnostics'): # Only makes sense to save labels for diagnostics.
                predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
                predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
                output_predict_file = os.path.join(training_args.output_dir, f"predictions-{test_task}.tsv")
                if trainer.is_world_process_zero():
                    with open(output_predict_file, "w") as writer:
                        logger.info(f"***** Predict results {task} *****")
                        writer.write("index\tlabel\tprediction\n")
                        for index, zipped in enumerate(zip(predictions, labels)):
                            pred_id, label_id = zipped
                            pred, label = label_list[pred_id], label_list[label_id]
                            writer.write(f"{index}\t{label}\t{pred}\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
