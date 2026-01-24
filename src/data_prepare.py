import torch
from transformers import AutoTokenizer
from datasets import load_dataset

from data_prepare_utils import tokenize_data_sample


def prepare_data_for_finetune(dataset_path: str, tokenizer: AutoTokenizer, seed: int, dataset_type: str,
                              is_contrastive=False, max_length: int = 4096, split_ratio: int = 0.2):
    """
    Load the dataset from a CSV file, tokenize the data samples, and return a transformer-compatible dataset.
    """
    dataset = load_dataset("csv", data_files=dataset_path, split="train")
    split_dataset = dataset.train_test_split(test_size=split_ratio, seed=seed)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    tokenized_train = train_dataset.map(
        tokenize_data_sample,
        batched=False,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": max_length,
            "dataset_type": dataset_type,
            "is_contrastive": is_contrastive
        },
        load_from_cache_file=False # can be set to True if you want
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
