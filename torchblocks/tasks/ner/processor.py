import json
import torch
import random
import logging
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from dataclasses import dataclass
from torchblocks.utils.common import check_object_type
from typing import Dict, List, Optional, Union, Any
from torchblocks.utils.tensor import sequence_padding
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def load_labels(label_path):
    labels = []
    with open(label_path, encoding='utf-8') as f:
        labels.extend(line.strip() for line in f)
    return labels


############################################ NER Dataset #######################################################

def process_dev_ner(example):
    return {"target": {(ent['label'], ent['start_offset'], ent['end_offset'] + 1, ent['entity']) for ent in example["entities"]}}


def get_ner_train_dev_dataset(
        data_dir,
        data_files,
        tokenizer,
        label_list,
        task_name='ner',
        train_max_seq_length=256,
        eval_max_seq_length=256,
        text_column_name="text",
        label_column_name="entities",
        is_chinese=True,
):
    ds = load_dataset(data_dir, data_files=data_files, cache_dir=data_dir)
    train_ds, dev_ds = ds['train'], ds['dev'].map(process_dev_ner)
    label2id = {v: i for i, v in enumerate(label_list)}

    def tokenize_and_align_train_labels(examples):
        # 英文文本使用空格分隔单词，BertTokenizer不对空格tokenize
        sentences = list(examples[text_column_name])
        if is_chinese:
            # 将中文文本的空格替换成其他字符，保证标签对齐
            sentences = [text.replace(" ", "-") for text in sentences]
        tokenized_inputs = tokenizer(
            sentences,
            max_length=train_max_seq_length,
            padding=False,
            truncation=True,
            return_token_type_ids=False,
        )
        labels = []
        for i, entity_list in enumerate(examples[label_column_name]):
            res = []
            for _ent in entity_list:
                try:
                    start = tokenized_inputs.char_to_token(i, _ent['start_offset'])
                    end = tokenized_inputs.char_to_token(i, _ent['end_offset'])
                except Exception:
                    logger.info("char_to_token error!")
                    continue
                if start is None or end is None:
                    continue
                res.append([start, end, label2id[_ent['label']]])
            labels.append(res)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def tokenize(examples):
        sentences = list(examples[text_column_name])
        if is_chinese:
            # 将中文文本的空格替换成其他字符，保证标签对齐
            sentences = [text.replace(" ", "-") for text in sentences]
        return tokenizer(sentences, max_length=eval_max_seq_length, padding=False, truncation=True,
                         return_offsets_mapping=True, return_token_type_ids=False)

    train_dataset = train_ds.map(
        tokenize_and_align_train_labels,
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


class NerDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=512, is_chinese=True):
        super().__init__()
        self.data = self.load_data(filename)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_chinese = is_chinese

    def __getitem__(self, index):
        data = self.data[index]
        sentence = data["text"].replace(" ", "-") if self.is_chinese else data['text']
        out = self.tokenizer(
            sentence,
            max_length=self.max_length,
            truncation=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
            padding=False,
        )
        out["text"] = data["text"]
        return out

    def __len__(self):
        return len(self.data)

    @staticmethod
    def load_data(filename):
        D = []
        with open(filename, encoding='utf-8') as f:
            D.extend(json.loads(line) for line in f)
        return D


def get_mrc_ner_train_dev_dataset(
        data_dir,
        data_files,
        tokenizer,
        label_list,
        task_name='mrc-ner',
        train_max_seq_length=256,
        eval_max_seq_length=256,
        text_column_name="text",
        label_column_name="entities",
        is_chinese=True,
):
    ds = load_dataset(data_dir, data_files=data_files, cache_dir=data_dir)
    train_ds, dev_ds = ds['train'], ds['dev'].map(process_dev_ner)

    def tokenize_and_align_train_labels(examples):
        check_object_type(object=label_list, check_type=dict, name='label_list')

        first_sentences = [list(label_list.values()) for _ in examples[text_column_name]]
        second_sentences = [[t] * len(label_list) for t in examples[text_column_name]]

        # flatten everthing
        first_sentences, second_sentences = sum(first_sentences, []), sum(second_sentences, [])
        if is_chinese:
            second_sentences = [text.replace(" ", "-") for text in second_sentences]

        tokenized_inputs = tokenizer(
            first_sentences,
            second_sentences,
            max_length=train_max_seq_length,
            padding=False,
            truncation='only_second',
            return_offsets_mapping=True,
        )

        all_label_dict = []
        for entity_list in examples[label_column_name]:
            label_dict = {k: [] for k in label_list.keys()}
            for _ent in entity_list:
                label_dict[_ent['label']].append((_ent['start_offset'], _ent['end_offset']))
            all_label_dict.extend(list(label_dict.values()))

        labels = []
        for i, lb in enumerate(all_label_dict):
            res = []
            input_ids = tokenized_inputs["input_ids"][i]
            offset_mapping = tokenized_inputs["offset_mapping"][i]
            # 区分prompt和text
            sequence_ids = tokenized_inputs.sequence_ids(i)
            for start, end in lb:
                # 找到token级别的index start
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1
                # 找到token级别的index end
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1
                # 检测答案是否在文本区间的外部
                if (offset_mapping[token_start_index][0] <= start) and (offset_mapping[token_end_index][1] >= end + 1):
                    while token_start_index < len(offset_mapping) and offset_mapping[token_start_index][0] <= start:
                        token_start_index += 1
                    while offset_mapping[token_end_index][1] >= (end + 1) and token_end_index > 0:
                        token_end_index -= 1
                    res.append((token_start_index - 1, token_end_index + 1))
            labels.append(res)
        tokenized_inputs["labels"] = labels
        return {k: [v[i: i + len(label_list)] for i in range(0, len(v), len(label_list))] for k, v in
                tokenized_inputs.items()}

    def tokenize(examples):
        check_object_type(object=label_list, check_type=dict, name='label_list')

        first_sentences = [list(label_list.values()) for _ in examples[text_column_name]]
        second_sentences = [[t] * len(label_list) for t in examples[text_column_name]]

        # flatten everthing
        first_sentences, second_sentences = sum(first_sentences, []), sum(second_sentences, [])
        if is_chinese:
            second_sentences = [text.replace(" ", "-") for text in second_sentences]

        tokenized_inputs = tokenizer(
            first_sentences,
            second_sentences,
            max_length=train_max_seq_length,
            padding=False,
            truncation='only_second',
            return_offsets_mapping=True,
        )
        return {k: [v[i: i + len(label_list)] for i in range(0, len(v), len(label_list))] for k, v in
                tokenized_inputs.items()}

    train_dataset = train_ds.map(
        tokenize_and_align_train_labels,
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


# dist_inputs
# https://github.com/ljynlp/W2NER/issues/17
dis2idx = torch.zeros(1000, dtype=torch.int64)
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


def get_w2ner_train_dev_dataset(
        data_dir,
        data_files,
        tokenizer,
        label_list,
        task_name='w2ner',
        train_max_seq_length=256,
        eval_max_seq_length=256,
        text_column_name="text",
        label_column_name="entities",
        is_chinese=True,
):
    ds = load_dataset(data_dir, data_files=data_files, cache_dir=data_dir)
    train_ds, dev_ds = ds['train'], ds['dev'].map(process_dev_ner)

    def tokenize_train(examples):
        check_object_type(object=label_list, check_type=list, name='label_list')
        label2id = {v: i for i, v in enumerate(label_list)}

        # 英文文本使用空格分隔单词，BertTokenizer不对空格tokenize
        sentences = list(examples[text_column_name])
        if is_chinese:
            # 将中文文本的空格替换成其他字符，保证标签对齐
            sentences = [text.replace(" ", "-") for text in sentences]
            
        input_keys = ["input_ids", "pieces2word", "dist_inputs", "grid_mask", "grid_labels"]
        encoded_inputs = {k: [] for k in input_keys}
        
        for sentence, label in zip(sentences, examples[label_column_name]): 
            tokens = [tokenizer.tokenize(word) for word in sentence[:train_max_seq_length - 2]]
            pieces = [piece for pieces in tokens for piece in pieces]
            _input_ids = tokenizer.convert_tokens_to_ids(pieces)
            _input_ids = np.array([tokenizer.cls_token_id] + _input_ids + [tokenizer.sep_token_id])
            
            length = len(tokens)
            # piece和word的对应关系
            _pieces2word = np.zeros((length, len(_input_ids)), dtype=np.bool)
            if tokenizer is not None:
                start = 0
                for i, pieces in enumerate(tokens):
                    if len(pieces) == 0:
                        continue
                    pieces = list(range(start, start + len(pieces)))
                    _pieces2word[i, pieces[0] + 1: pieces[-1] + 2] = 1
                    start += len(pieces)
            
            # 相对距离
            _dist_inputs = np.zeros((length, length), dtype=np.int)
            for k in range(length):
                _dist_inputs[k, :] += k
                _dist_inputs[:, k] -= k

            for i in range(length):
                for j in range(length):
                    if _dist_inputs[i, j] < 0:
                        _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                    else:
                        _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
            _dist_inputs[_dist_inputs == 0] = 19
            
            # 标签
            _grid_labels = np.zeros((length, length), dtype=np.int)
            _grid_mask = np.ones((length, length), dtype=np.bool)
            
            for entity in label:
                if "index" in entity:
                    index = entity["index"]
                else:
                    _start, _end, _type = entity["start_offset"], entity["end_offset"] + 1, entity["label"]
                    index = list(range(_start, _end))
                    
                if index[-1] >= train_max_seq_length - 2:
                    continue
                
                for i in range(len(index)):
                    if i + 1 >= len(index):
                        break
                    _grid_labels[index[i], index[i + 1]] = 1
                _grid_labels[index[-1], index[0]] = label2id[_type] + 2
        
            for k, v in zip(input_keys, [_input_ids, _pieces2word, _dist_inputs, _grid_mask, _grid_labels]):
                encoded_inputs[k].append(list(v))
                
        return encoded_inputs
    
    def tokenize(examples):
        # 英文文本使用空格分隔单词，BertTokenizer不对空格tokenize
        sentences = list(examples[text_column_name])
        if is_chinese:
            # 将中文文本的空格替换成其他字符，保证标签对齐
            sentences = [text.replace(" ", "-") for text in sentences]
            
        input_keys = ["input_ids", "pieces2word", "dist_inputs", "grid_mask"]
        encoded_inputs = {k: [] for k in input_keys}
        
        for sentence in sentences: 
            tokens = [tokenizer.tokenize(word) for word in sentence[:train_max_seq_length - 2]]
            pieces = [piece for pieces in tokens for piece in pieces]
            _input_ids = tokenizer.convert_tokens_to_ids(pieces)
            _input_ids = np.array([tokenizer.cls_token_id] + _input_ids + [tokenizer.sep_token_id])
            
            length = len(tokens)
            # piece和word的对应关系
            _pieces2word = np.zeros((length, len(_input_ids)), dtype=np.bool)
            if tokenizer is not None:
                start = 0
                for i, pieces in enumerate(tokens):
                    if len(pieces) == 0:
                        continue
                    pieces = list(range(start, start + len(pieces)))
                    _pieces2word[i, pieces[0] + 1: pieces[-1] + 2] = 1
                    start += len(pieces)
            
            # 相对距离
            _dist_inputs = np.zeros((length, length), dtype=np.int)
            for k in range(length):
                _dist_inputs[k, :] += k
                _dist_inputs[:, k] -= k

            for i in range(length):
                for j in range(length):
                    if _dist_inputs[i, j] < 0:
                        _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                    else:
                        _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
            _dist_inputs[_dist_inputs == 0] = 19
            
            _grid_mask = np.ones((length, length), dtype=np.bool)
        
            for k, v in zip(input_keys, [_input_ids, _pieces2word, _dist_inputs, _grid_mask]):
                encoded_inputs[k].append(list(v))
                
        return encoded_inputs

    train_dataset = train_ds.map(
        tokenize_train,
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


############################################ NER Collator #######################################################

ignore_list = ["offset_mapping", "text", "target"]


@dataclass
class DataCollatorForSpanNer:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = ([feature.pop("labels") for feature in features] if "labels" in features[0].keys() else None)
        new_features = [{k: v for k, v in f.items() if k not in ignore_list} for f in features]

        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:  # for test
            if "text" in features[0].keys():
                batch["texts"] = [feature.pop("text") for feature in features]
            if "offset_mapping" in features[0].keys():
                batch["offset_mapping"] = [feature.pop("offset_mapping") for feature in features]
            if "target" in features[0].keys():
                batch['target'] = [{tuple(t) for t in feature.pop("target")} for feature in features]
            return batch

        batch_start_positions = torch.zeros_like(batch["input_ids"])
        batch_end_positions = torch.zeros_like(batch["input_ids"])
        for i, lb in enumerate(labels):
            for start, end, tag in lb:
                batch_start_positions[i, start] = tag + 1
                batch_end_positions[i, end] = tag + 1

        batch['start_positions'] = batch_start_positions
        batch['end_positions'] = batch_end_positions
        return batch


@dataclass
class DataCollatorForCRF:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = ([feature.pop("labels") for feature in features] if "labels" in features[0].keys() else None)
        new_features = [{k: v for k, v in f.items() if k not in ignore_list} for f in features]

        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:  # for test
            if "text" in features[0].keys():
                batch["texts"] = [feature.pop("text") for feature in features]
            if "offset_mapping" in features[0].keys():
                batch["offset_mapping"] = [feature.pop("offset_mapping") for feature in features]
            if "target" in features[0].keys():
                batch['target'] = [{tuple(t) for t in feature.pop("target")} for feature in features]
            return batch

        batch_label_ids = torch.zeros_like(batch["input_ids"])
        for i, lb in enumerate(labels):
            for start, end, tag in lb:
                batch_label_ids[i, start] = tag + 1  # B
                batch_label_ids[i, start + 1: end + 1] = tag + self.num_labels + 1  # I

        batch['labels'] = batch_label_ids
        return batch


@dataclass
class DataCollatorForCascadeCRF:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = ([feature.pop("labels") for feature in features] if "labels" in features[0].keys() else None)
        new_features = [{k: v for k, v in f.items() if k not in ignore_list} for f in features]

        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:  # for test
            if "text" in features[0].keys():
                batch["texts"] = [feature.pop("text") for feature in features]
            if "offset_mapping" in features[0].keys():
                batch["offset_mapping"] = [feature.pop("offset_mapping") for feature in features]
            if "target" in features[0].keys():
                batch['target'] = [{tuple(t) for t in feature.pop("target")} for feature in features]
            return batch

        batch_entity_labels = torch.zeros_like(batch["input_ids"])
        batch_entity_ids, batch_labels = [], []
        for i, lb in enumerate(labels):
            entity_ids, label = [], []
            for start, end, tag in lb:
                batch_entity_labels[i, start] = 1  # B
                batch_entity_labels[i, start + 1: end + 1] = 2  # I
                entity_ids.append([start, end])
                label.append(tag + 1)
            if not entity_ids:
                entity_ids.append([0, 0])
                label.append(0)
            batch_entity_ids.append(entity_ids)
            batch_labels.append(label)

        batch['entity_labels'] = batch_entity_labels
        batch['entity_ids'] = torch.from_numpy(sequence_padding(batch_entity_ids))
        batch['labels'] = torch.from_numpy(sequence_padding(batch_labels))
        return batch


@dataclass
class DataCollatorForGlobalPointer:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_labels: Optional[int] = None
    is_sparse: Optional[bool] = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = ([feature.pop("labels") for feature in features] if "labels" in features[0].keys() else None)
        new_features = [{k: v for k, v in f.items() if k not in ignore_list} for f in features]

        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:  # for test
            if "text" in features[0].keys():
                batch["texts"] = [feature.pop("text") for feature in features]
            if "offset_mapping" in features[0].keys():
                batch["offset_mapping"] = [feature.pop("offset_mapping") for feature in features]
            if "target" in features[0].keys():
                batch['target'] = [{tuple(t) for t in feature.pop("target")} for feature in features]
            return batch

        bs, seqlen = batch["input_ids"].shape
        if self.is_sparse:
            batch_labels = []
            for lb in labels:
                label = [set() for _ in range(self.num_labels)]
                for start, end, tag in lb:
                    label[tag].add((start, end))
                for l in label:
                    if not l:  # 至少要有一个标签
                        l.add((0, 0))  # 如果没有则用0填充 
                label = sequence_padding([list(l) for l in label])
                batch_labels.append(label)
            batch_labels = torch.from_numpy(sequence_padding(batch_labels, seq_dims=2))
        else:
            batch_labels = torch.zeros(bs, self.num_labels, seqlen, seqlen, dtype=torch.long)
            for i, lb in enumerate(labels):
                for start, end, tag in lb:
                    batch_labels[i, tag, start, end] = 1  # 0为"O"

        batch["labels"] = batch_labels
        return batch


@dataclass
class DataCollatorForTPLinkerPlusNer:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = ([feature.pop("labels") for feature in features] if "labels" in features[0].keys() else None)
        new_features = [{k: v for k, v in f.items() if k not in ignore_list} for f in features]

        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:  # for test
            if "text" in features[0].keys():
                batch["texts"] = [feature.pop("text") for feature in features]
            if "offset_mapping" in features[0].keys():
                batch["offset_mapping"] = [feature.pop("offset_mapping") for feature in features]
            if "target" in features[0].keys():
                batch['target'] = [{tuple(t) for t in feature.pop("target")} for feature in features]
            return batch

        bs, seqlen = batch["input_ids"].shape
        mask = torch.triu(torch.ones(seqlen, seqlen), diagonal=0).bool()
        batch_shaking_tag = torch.zeros(bs, seqlen, seqlen, self.num_labels, dtype=torch.long)

        for i, lb in enumerate(labels):
            for start, end, tag in lb:
                batch_shaking_tag[i, start, end, tag] = 1

        batch["labels"] = batch_shaking_tag.masked_select(mask[None, :, :, None]).reshape(bs, -1, self.num_labels)
        return batch


@dataclass
class DataCollatorForMRCNer:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        flattened_features = [
            [{k: v[i] for k, v in feature.items() if k not in ignore_list} for i in range(self.num_labels)] for feature
            in features]
        flattened_features = sum(flattened_features, [])
        labels = [feature.pop("labels") for feature in flattened_features] if "labels" in flattened_features[
            0].keys() else None

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:  # for test
            if "text" in features[0].keys():
                batch["texts"] = [feature["text"] for feature in features for _ in range(self.num_labels)]
            if "offset_mapping" in features[0].keys():
                batch["offset_mapping"] = [feature["offset_mapping"][i] for feature in features for i in
                                           range(self.num_labels)]
            if "target" in features[0].keys():
                batch['target'] = [{tuple(t) for t in feature.pop("target")} for feature in features]
            return batch

        batch_start_positions = torch.zeros_like(batch["input_ids"])
        batch_end_positions = torch.zeros_like(batch["input_ids"])
        for i, lb in enumerate(labels):
            for start, end in lb:
                batch_start_positions[i, start] = 1
                batch_end_positions[i, end] = 1

        batch['start_positions'] = batch_start_positions
        batch['end_positions'] = batch_end_positions
        return batch


@dataclass
class DataCollatorForLEARNer:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_annotations: Optional[List[str]] = None
    nested: Optional[bool] = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = ([feature.pop("labels") for feature in features] if "labels" in features[0].keys() else None)
        new_features = [{k: v for k, v in f.items() if k not in ignore_list} for f in features]

        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        label_batch = self.tokenizer(
            list(self.label_annotations),
            padding=self.padding,
            truncation=True,
            max_length=64,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        label_batch = {f"label_{k}": v for k, v in label_batch.items()}

        if labels is None:  # for test
            if "text" in features[0].keys():
                batch["texts"] = [feature.pop("text") for feature in features]
            if "offset_mapping" in features[0].keys():
                batch["offset_mapping"] = [feature.pop("offset_mapping") for feature in features]
            if "target" in features[0].keys():
                batch['target'] = [{tuple(t) for t in feature.pop("target")} for feature in features]
            return {**batch, **label_batch}

        bs, seqlen = batch["input_ids"].shape
        num_labels = len(self.label_annotations)
        batch_start_labels = torch.zeros(bs, seqlen, num_labels, dtype=torch.long)
        batch_end_labels = torch.zeros(bs, seqlen, num_labels, dtype=torch.long)
        if self.nested:
            batch_span_labels = torch.zeros(bs, seqlen, seqlen, num_labels, dtype=torch.long)

        for i, lb in enumerate(labels):
            for start, end, tag in lb:
                batch_start_labels[i, start, tag] = 1
                batch_end_labels[i, end, tag] = 1
                if self.nested:
                    batch_span_labels[i, start, end, tag] = 1

        batch["start_labels"] = batch_start_labels
        batch["end_labels"] = batch_end_labels
        if self.nested:
            batch["span_labels"] = batch_span_labels
        return {**batch, **label_batch}


@dataclass
class DataCollatorForW2Ner:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = ([feature.pop("grid_labels") for feature in features] if "grid_labels" in features[0].keys() else None)
        
        input_ids = [feature.pop("input_ids") for feature in features]
        input_ids = torch.from_numpy(sequence_padding(input_ids))
        
        pieces2word = [feature.pop("pieces2word") for feature in features]
        input_lengths = torch.tensor([len(i) for i in pieces2word], dtype=torch.long)
        
        max_wordlen = torch.max(input_lengths).item()
        max_pieces_len = max([x.shape[0] for x in input_ids])
        
        batch_size = input_ids.shape[0]
        sub_mat = torch.zeros(batch_size, max_wordlen, max_pieces_len, dtype=torch.long)
        pieces2word = self.fill(pieces2word, sub_mat)
        
        dist_mat = torch.zeros(batch_size, max_wordlen, max_wordlen, dtype=torch.long)
        dist_inputs = [feature.pop("dist_inputs") for feature in features]
        dist_inputs = self.fill(dist_inputs, dist_mat)
        
        mask_mat = torch.zeros(batch_size, max_wordlen, max_wordlen, dtype=torch.long)
        grid_mask = [feature.pop("grid_mask") for feature in features]
        grid_mask = self.fill(grid_mask, mask_mat)
        
        batch = {
            "input_ids": input_ids,
            "dist_inputs": dist_inputs,
            "pieces2word": pieces2word,
            "grid_mask": grid_mask,
            "input_lengths": input_lengths,
        }
        
        if labels is None:  # for test
            if "text" in features[0].keys():
                batch["texts"] = [feature.pop("text") for feature in features]
            if "target" in features[0].keys():
                batch['target'] = [{tuple(t) for t in feature.pop("target")} for feature in features]
            return batch
            
        labels_mat = torch.zeros(batch_size, max_wordlen, max_wordlen, dtype=torch.long)
        labels = self.fill(labels, labels_mat)
        batch["grid_labels"] = labels
        
        return batch
    
    @staticmethod
    def fill(data, new_data):
        for i, d in enumerate(data):
            new_data[i, :len(d), :len(d[0])] = torch.tensor(d, dtype=torch.long)
        return new_data
            

NER_COLLATOR_MAP = {
    "crf": DataCollatorForCRF,
    "cascade-crf": DataCollatorForCascadeCRF,
    "softmax": DataCollatorForCRF,
    "span": DataCollatorForSpanNer,
    "global-pointer": DataCollatorForGlobalPointer,
    "mrc": DataCollatorForMRCNer,
    "tplinker": DataCollatorForTPLinkerPlusNer,
    "lear": DataCollatorForLEARNer,
    "w2ner": DataCollatorForW2Ner,
}


def get_auto_ner_collator(model_name: str = "crf"):
    return NER_COLLATOR_MAP[model_name]
