import torch
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


############################################ RE Dataset #######################################################

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    return next((i for i in range(len(sequence)) if sequence[i: i + n] == pattern), -1)


def judge(example):
    spo_list = []
    for spo in example["spo_list"]:
        sub = search(spo["subject"], example["text"])
        obj = search(spo["object"], example["text"])
        if sub == -1 or obj == -1:
            continue
        else:
            spo_list.append([1])
    return len(spo_list) > 0


def process_train_re(ds, predicate2id):
    def convert(example):
        spo_list = []
        for spo in example["spo_list"]:
            sub = search(spo["subject"], example["text"])
            pre = predicate2id[spo["predicate"]]
            obj = search(spo["object"], example["text"])
            if sub == -1 or obj == -1:
                continue
            else:
                spo_list.append(
                    [
                        sub,
                        sub + len(spo["subject"]) - 1,
                        pre,
                        obj,
                        obj + len(spo["object"]) - 1,
                    ]
                )

        assert spo_list
        return {"text": example["text"], "spo_list": spo_list}

    return ds.filter(judge).map(convert)


def process_dev_re(example):
    triplet = [[spo["subject"], spo["predicate"], spo["object"], ] for spo in example["spo_list"]]
    return {"text": example["text"], "target": triplet}


def get_re_train_dev_dataset(
        data_dir,
        data_files,
        tokenizer,
        predicates,
        has_offset=False,
        task_name='re',
        train_max_seq_length=256,
        eval_max_seq_length=256,
        text_column_name="text",
        label_column_name="spo_list",
        is_chinese=True,
):
    ds = load_dataset(data_dir, data_files=data_files, cache_dir=data_dir)
    predicate2id = {v: i for i, v in enumerate(predicates)}
    train_ds = process_train_re(ds["train"], predicate2id=predicate2id) if not has_offset else ds['train']
    dev_ds = ds['dev'].map(process_dev_re)

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
        for i, spo_list in enumerate(examples[label_column_name]):
            spo = []
            for _sh, _st, p, _oh, _ot in spo_list:
                try:
                    sh = tokenized_inputs.char_to_token(i, _sh)
                    oh = tokenized_inputs.char_to_token(i, _oh)
                    st = tokenized_inputs.char_to_token(i, _st)
                    ot = tokenized_inputs.char_to_token(i, _ot)
                except Exception:
                    logger.info("char_to_token error!")
                    continue
                if sh is None or oh is None or st is None or ot is None:
                    continue
                spo.append([sh, st, p, oh, ot])
            labels.append(spo)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def tokenize(examples):
        # 英文文本使用空格分隔单词，BertTokenizer不对空格tokenize
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


############################################ RE Collator #######################################################

ignore_list = ["offset_mapping", "text", "target", "spn_labels"]


@dataclass
class DataCollatorForCasRel:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_predicates: Optional[int] = None

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
        batch_subject_labels = torch.zeros(bs, seqlen, 2, dtype=torch.long)
        batch_object_labels = torch.zeros(bs, seqlen, self.num_predicates, 2, dtype=torch.long)
        batch_subject_ids = torch.zeros(bs, 2, dtype=torch.long)

        for i, lb in enumerate(labels):
            spoes = {}
            for sh, st, p, oh, ot in lb:
                if (sh, st) not in spoes:
                    spoes[(sh, st)] = []
                spoes[(sh, st)].append((oh, ot, p))
            if spoes:
                for s in spoes:
                    batch_subject_labels[i, s[0], 0] = 1
                    batch_subject_labels[i, s[1], 1] = 1
                # 随机选一个subject
                subject_ids = random.choice(list(spoes.keys()))
                batch_subject_ids[i, 0] = subject_ids[0]
                batch_subject_ids[i, 1] = subject_ids[1]
                for o in spoes.get(subject_ids, []):
                    batch_object_labels[i, o[0], o[2], 0] = 1
                    batch_object_labels[i, o[1], o[2], 1] = 1

        batch["subject_labels"] = batch_subject_labels
        batch["object_labels"] = batch_object_labels
        batch["subject_ids"] = batch_subject_ids
        return batch


@dataclass
class DataCollatorForSPN:
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

        spn_labels = []
        for lb in labels:
            spn_label = {
                "relation": [],
                "head_start_index": [],
                "head_end_index": [],
                "tail_start_index": [],
                "tail_end_index": []
            }
            for sh, st, p, oh, ot in lb:
                spn_label["relation"].append(p)
                spn_label["head_start_index"].append(sh)
                spn_label["head_end_index"].append(st)
                spn_label["tail_start_index"].append(oh)
                spn_label["tail_end_index"].append(ot)
            spn_labels.append({k: torch.tensor(v, dtype=torch.long) for k, v in spn_label.items()})

        batch['spn_labels'] = spn_labels
        return batch


@dataclass
class DataCollatorForGPLinker:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_predicates: Optional[int] = None

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

        bs = batch["input_ids"].size(0)
        max_spo_num = max(len(lb) for lb in labels)
        batch_entity_labels = torch.zeros(bs, 2, max_spo_num, 2, dtype=torch.long)
        batch_head_labels = torch.zeros(bs, self.num_predicates, max_spo_num, 2, dtype=torch.long)
        batch_tail_labels = torch.zeros(bs, self.num_predicates, max_spo_num, 2, dtype=torch.long)

        for i, lb in enumerate(labels):
            for spidx, (sh, st, p, oh, ot) in enumerate(lb):
                batch_entity_labels[i, 0, spidx, :] = torch.tensor([sh, st])
                batch_entity_labels[i, 1, spidx, :] = torch.tensor([oh, ot])
                batch_head_labels[i, p, spidx, :] = torch.tensor([sh, oh])
                batch_tail_labels[i, p, spidx, :] = torch.tensor([st, ot])

        batch["entity_labels"] = batch_entity_labels
        batch["head_labels"] = batch_head_labels
        batch["tail_labels"] = batch_tail_labels
        return batch


@dataclass
class DataCollatorForTPLinkerPlus:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_predicates: Optional[int] = None

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
        num_tag = self.num_predicates * 4 + 1
        batch_shaking_tag = torch.zeros(bs, seqlen, seqlen, num_tag, dtype=torch.long)

        for i, lb in enumerate(labels):
            for sh, st, p, oh, ot in lb:
                # SH2OH
                batch_shaking_tag[i, sh, oh, p] = 1
                # OH2SH
                batch_shaking_tag[i, oh, sh, p + self.num_predicates] = 1
                # ST2OT
                batch_shaking_tag[i, st, ot, p + self.num_predicates * 2] = 1
                # OT2ST
                batch_shaking_tag[i, ot, st, p + self.num_predicates * 3] = 1
                # EH2ET
                batch_shaking_tag[i, sh, st, -1] = 1
                batch_shaking_tag[i, oh, ot, -1] = 1

        batch["labels"] = batch_shaking_tag.masked_select(mask[None, :, :, None]).reshape(bs, -1, num_tag)
        return batch


@dataclass
class DataCollatorForGRTE:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_predicates: Optional[int] = None
    label2id: Optional[dict] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # sourcery skip: low-code-quality
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
        batch_labels = torch.zeros(bs, seqlen, seqlen, self.num_predicates, dtype=torch.long)
        for i, lb in enumerate(labels):
            spoes = {}
            for sh, st, p, oh, ot in lb:
                if (sh, st) not in spoes:
                    spoes[(sh, st)] = []
                spoes[(sh, st)].append((oh, ot, p))
            if spoes:
                for s in spoes:
                    sh, st = s
                    for oh, ot, p in spoes[(sh, st)]:
                        if sh == st and oh == ot:
                            batch_labels[i, sh, oh, p] = self.label2id['SS']
                        elif sh != st and oh == ot:
                            batch_labels[i, sh, oh, p] = self.label2id['MSH']
                            batch_labels[i, st, oh, p] = self.label2id['MST']
                        elif sh == st:
                            batch_labels[i, sh, oh, p] = self.label2id['SMH']
                            batch_labels[i, sh, ot, p] = self.label2id['SMT']
                        else:
                            batch_labels[i, sh, oh, p] = self.label2id['MMH']
                            batch_labels[i, st, ot, p] = self.label2id['MMT']

        batch["labels"] = batch_labels
        return batch


@dataclass
class DataCollatorForPRGC:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_predicates: Optional[int] = None
    negative_ratio: Optional[int] = None

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

        new_batch = {
            "input_ids": [],
            "attention_mask": [],
            "seq_labels": [],
            "corres_labels": [],
            "potential_rels": [],
            "rel_labels": []
        }

        seqlen = batch["input_ids"].shape[1]
        for i, lb in enumerate(labels):
            corres_label = torch.zeros(seqlen, seqlen, dtype=torch.long)
            spoes = {}
            for sh, st, p, oh, ot in lb:
                corres_label[sh, oh] = 1
                if p not in spoes:
                    spoes[p] = []
                spoes[p].append((sh, st, oh, ot))

            # rel one-hot label
            rel_label = torch.zeros(self.num_predicates, dtype=torch.long)
            for p in spoes:
                rel_label[p] = 1

            # positive samples
            for p in spoes:
                # subject, object B-I-O label
                seq_label = torch.zeros(2, seqlen, dtype=torch.long)
                for sh, st, oh, ot in spoes[p]:
                    seq_label[0, sh] = 1  # B-ENT
                    seq_label[0, sh + 1: st + 1] = 2  # I-ENT
                    seq_label[1, oh] = 1  # B-ENT
                    seq_label[1, oh + 1: ot + 1] = 2  # I-ENT
                new_batch["input_ids"].append(batch["input_ids"][i])
                new_batch["attention_mask"].append(batch["attention_mask"][i])
                new_batch["rel_labels"].append(rel_label)
                new_batch["seq_labels"].append(seq_label)
                new_batch["corres_labels"].append(corres_label)
                new_batch["potential_rels"].append(p)

            # negtive samples
            neg_rels = set(range(self.num_predicates)).difference(set(spoes.keys()))
            if neg_rels:
                neg_rels = random.sample(neg_rels, k=min(len(neg_rels), self.negative_ratio))
            for neg_rel in neg_rels:
                # subject, object B-I-O label
                seq_label = torch.zeros(2, seqlen, dtype=torch.long)
                new_batch["input_ids"].append(batch["input_ids"][i])
                new_batch["attention_mask"].append(batch["attention_mask"][i])
                new_batch["rel_labels"].append(rel_label)
                new_batch["seq_labels"].append(seq_label)
                new_batch["corres_labels"].append(corres_label)
                new_batch["potential_rels"].append(neg_rel)

        return {k: torch.stack(v) for k, v in new_batch.items()}


@dataclass
class DataCollatorForPFN:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_predicates: Optional[int] = None

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
        batch_entity_labels = torch.zeros(bs, 2, seqlen, seqlen, dtype=torch.long)
        batch_head_labels = torch.zeros(bs, self.num_predicates, seqlen, seqlen, dtype=torch.long)
        batch_tail_labels = torch.zeros(bs, self.num_predicates, seqlen, seqlen, dtype=torch.long)

        for i, lb in enumerate(labels):
            for sh, st, p, oh, ot in lb:
                batch_entity_labels[i, 0, sh, st] = 1
                batch_entity_labels[i, 1, oh, ot] = 1
                batch_head_labels[i, p, sh, oh] = 1
                batch_tail_labels[i, p, st, ot] = 1

        batch["entity_labels"] = batch_entity_labels
        batch["head_labels"] = batch_head_labels
        batch["tail_labels"] = batch_tail_labels
        return batch


RE_COLLATOR_MAP = {
    "casrel": DataCollatorForCasRel,
    "gplinker": DataCollatorForGPLinker,
    "tplinker": DataCollatorForTPLinkerPlus,
    "grte": DataCollatorForGRTE,
    "spn": DataCollatorForSPN,
    "prgc": DataCollatorForPRGC,
    "pfn": DataCollatorForPFN,
}


def get_auto_re_collator(model_name: str = "gplinker"):
    return RE_COLLATOR_MAP[model_name]
