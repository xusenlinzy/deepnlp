import random
import logging
from datasets import load_dataset
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def load_labels(label_path):
    labels = []
    with open(label_path, encoding='utf-8') as f:
        labels.extend(line.strip() for line in f)
    return labels


def get_tc_train_dev_dataset(
        data_dir,
        data_files,
        tokenizer,
        label_list,
        task_name='tc',
        train_max_seq_length=256,
        eval_max_seq_length=256,
        text_column_name="text",
        label_column_name="label",
):
    ds = load_dataset(data_dir, data_files=data_files, cache_dir=data_dir)
    label2id = {v: i for i, v in enumerate(label_list)}

    def transform_label(example):
        return {"text": example["text"], "label": label2id[example[label_column_name]]}

    train_ds, dev_ds = ds['train'].map(transform_label), ds['dev'].map(transform_label)

    def tokenize(examples, max_length=256):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=max_length,
            padding=False,
            truncation=True,
            return_token_type_ids=False,
        )
        return tokenized_inputs

    def tokenize_train(examples):
        return tokenize(examples, max_length=train_max_seq_length)

    def tokenize_dev(examples):
        return tokenize(examples, max_length=eval_max_seq_length)

    train_dataset = train_ds.map(
        tokenize_train,
        batched=True,
        remove_columns=[text_column_name],
        desc="Running tokenizer on train dataset",
        new_fingerprint=f"train-{train_max_seq_length}-{task_name}",
    )
    dev_dataset = dev_ds.map(
        tokenize_dev,
        batched=True,
        remove_columns=[text_column_name],
        desc="Running tokenizer on dev dataset",
        new_fingerprint=f"dev-{eval_max_seq_length}-{task_name}",
    )

    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set:")
        for k, v in train_dataset[index].items():
            logger.info(f"{k} = {v}")

    return train_dataset, dev_dataset


############################################ TC Collator #######################################################

@dataclass
class DataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch
