"""
A script that contains data preparation functions for fine-tuning the TRAIL model.
It loads the dataset, tokenizes the data samples, and returns a transformer-compatible dataset.

Author: Zhixi Chen
Date: 2025/06/18
"""

from transformers import AutoTokenizer
from datasets import load_dataset

from .data_prepare_utils import tokenize_data_sample


def prepare_data_for_finetune(dataset_path: str, tokenizer: AutoTokenizer, seed: int, dataset_type: str,
                              is_contrastive=False, max_length: int = 4096, split_ratio: float = 0.2):
    """
    Load the dataset from a CSV file, tokenize the data samples, and return a transformer-compatible dataset.
    """
    ##### 1. Load and tokenize the dataset #####
    dataset = load_dataset("csv", data_files=dataset_path, split="train")
    split_dataset = dataset.train_test_split(test_size=split_ratio, seed=seed)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']

    # `tokenize_data_sample` returns None when the truncated sample loses <exp_start> or <exp_end>.
    # `map` can not drop a sample, so these samples are removed first.
    def has_explanation_tokens(sample):
        return tokenize_data_sample(sample, tokenizer, max_length, dataset_type, is_contrastive) is not None

    train_dataset = train_dataset.filter(has_explanation_tokens, load_from_cache_file=False)
    val_dataset = val_dataset.filter(has_explanation_tokens, load_from_cache_file=False)

    tokenized_train = train_dataset.map(
        tokenize_data_sample,
        batched=False,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": max_length,
            "dataset_type": dataset_type,
            "is_contrastive": is_contrastive
        },
        load_from_cache_file=False
    )

    tokenized_val = val_dataset.map(
        tokenize_data_sample,
        batched=False,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": max_length,
            "dataset_type": dataset_type,
            "is_contrastive": is_contrastive
        },
        load_from_cache_file=False
    )

    return tokenized_train, tokenized_val
