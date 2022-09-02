import random
import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_labels(label_path):
    labels = []
    with open(label_path, encoding='utf-8') as f:
        labels.extend(line.strip() for line in f)
    return labels


############################################ MultipleChoice #######################################################

def get_mc_train_dev_dataset(
        data_dir,
        data_files,
        tokenizer,
        label_list,
        task_name='multiple-choice',
        train_max_seq_length=256,
        eval_max_seq_length=256,
        context_column_name="context",
        question_column_name="question",
        choices_column_name="choices",
        label_column_name="answer"
):
    ds = load_dataset(data_dir, data_files=data_files, cache_dir=data_dir)
    label2id = {v: i for i, v in enumerate(label_list)}

    def transform_labels(example):
        return {'label': label2id[example[label_column_name]]}

    train_ds, dev_ds = ds['train'].map(transform_labels), ds['dev'].map(transform_labels)

    def tokenize(examples):
        first_sentences = [[f"{q} {c}" for c in examples[choices_column_name]] for q in examples[question_column_name]]
        second_sentences = [[t] * len(label2id) for t in examples[context_column_name]]
        
        # flatten everything
        first_sentences, second_sentences = sum(first_sentences, []), sum(second_sentences, [])
        tokenized_inputs = tokenizer(
            first_sentences,
            second_sentences,
            max_length=train_max_seq_length,
            padding=False,
            truncation='only_second'
        )
        return {k: [v[i: i + len(label2id)] for i in range(0, len(v), len(label2id))] for k, v in
                tokenized_inputs.items()}

    train_dataset = train_ds.map(
        tokenize,
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Running tokenizer on train dataset",
        new_fingerprint=f"train-{train_max_seq_length}-{task_name}",
    )
    dev_dataset = dev_ds.map(
        tokenize,
        batched=True,
        remove_columns=[label_column_name],
        desc="Running tokenizer on dev dataset",
        new_fingerprint=f"dev-{eval_max_seq_length}-{task_name}",
    )

    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set:")
        for k, v in train_dataset[index].items():
            logger.info(f"{k} = {v}")

    return train_dataset, dev_dataset
