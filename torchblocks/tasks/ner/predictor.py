import torch
import itertools
import numpy as np
from collections import defaultdict
from typing import List, Union
from transformers import BertTokenizerFast
from .auto import get_auto_ner_model
from .processor import dis2idx, DataCollatorForW2Ner
from ..predictor_base import PredictorBase
from ...utils.common import auto_splitter
from ..uie.utils import logger, tqdm


def set2json(labels) -> dict:
    """ 将实体集合根据实体类型转换为字典    
    """
    res = {}
    for _type, _start, _end, _ent in labels:
        dic = {"start": _start, "end": _end, "text": _ent}
        if _type not in res:
            res[_type] = [dic]
        else:
            res[_type].append(dic)
    return res


class NERPredictor(PredictorBase):
    """
    A class for NER task predictor.
    """

    @torch.no_grad()
    def predict(
            self,
            text: Union[str, List[str]],
            batch_size: int = 64,
            max_length: int = 512,
            return_dict: bool = True,
    ) -> Union[dict, List[dict]]:
        # sourcery skip: boolean-if-exp-identity, remove-unnecessary-cast

        single_sentence = False
        if isinstance(text, str):
            text = [text]
            single_sentence = True

        text = [t.replace(" ", "-") for t in text]  # 防止空格导致位置预测偏移

        output_list = []
        total_batch = len(text) // batch_size + (1 if len(text) % batch_size > 0 else 0)
        for batch_id in tqdm(range(total_batch), desc="Predicting"):
            batch_text = text[batch_id * batch_size: (batch_id + 1) * batch_size]

            inputs = self.tokenizer(
                batch_text,
                max_length=max_length,
                truncation=True,
                return_offsets_mapping=True,
                padding=True,
                return_tensors="pt",
            )

            inputs['texts'] = batch_text
            inputs["offset_mapping"] = inputs["offset_mapping"].tolist()

            inputs = self.build_batch_inputs(inputs)
            outputs = self.model(**inputs)
            output_list.extend(outputs['predictions'])

        if not return_dict:
            return output_list[0] if single_sentence else output_list
        else:
            return set2json(output_list[0]) if single_sentence else [set2json(o) for o in output_list]


class PromptNERPredictor(PredictorBase):
    """ 基于prompt的predictor
    """

    def __init__(self, schema2prompt: dict = None, **kwargs):
        self.schema2prompt = schema2prompt
        super().__init__(**kwargs)
        self.model.config.label_list = schema2prompt

    @torch.no_grad()
    def predict(
            self,
            text: Union[str, List[str]],
            batch_size: int = 8,
            max_length: int = 512,
            return_dict: bool = True,
    ) -> Union[dict, List[dict]]:

        if isinstance(text, str):
            return self.single_sample_predict(text, max_length, return_dict)
        else:
            return [self.single_sample_predict(sent, max_length, return_dict) for sent in text]

    def single_sample_predict(self, text: str, max_length: int = 512, return_dict: bool = True):
        text = text.replace(" ", "-")  # 防止空格导致位置预测偏移
        first_sentences = list(self.schema2prompt.values())
        second_sentences = [text] * len(self.schema2prompt)

        inputs = self.tokenizer(
            first_sentences,
            second_sentences,
            max_length=max_length,
            padding=True,
            truncation='only_second',
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        inputs['texts'] = second_sentences
        inputs["offset_mapping"] = inputs["offset_mapping"].tolist()

        inputs = self.build_batch_inputs(inputs)
        outputs = self.model(**inputs)['predictions']

        return set2json(outputs[0]) if return_dict else outputs[0]


class LearNERPredictor(PredictorBase):
    """ 基于LEAR的predictor
    """

    def __init__(self, schema2prompt: dict = None, **kwargs):
        self.schema2prompt = schema2prompt
        super().__init__(**kwargs)

    @torch.no_grad()
    def predict(
            self,
            text: Union[str, List[str]],
            batch_size: int = 8,
            max_length: int = 512,
            return_dict: bool = True,
    ) -> Union[dict, List[dict]]:

        single_sentence = False
        if isinstance(text, str):
            text = [text]
            single_sentence = True

        text = [t.replace(" ", "-") for t in text]  # 防止空格导致位置预测偏移
        output_list = []

        label_annotations = list(self.schema2prompt.values())
        label_inputs = self.tokenizer(
            label_annotations,
            padding=True,
            truncation=True,
            max_length=64,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        label_inputs = {f"label_{k}": v for k, v in label_inputs.items()}

        total_batch = len(text) // batch_size + (1 if len(text) % batch_size > 0 else 0)
        for batch_id in tqdm(range(total_batch), desc="Predicting"):
            batch_text = text[batch_id * batch_size: (batch_id + 1) * batch_size]
            inputs = self.tokenizer(
                batch_text,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_offsets_mapping=True,
                return_tensors="pt"
            )

            inputs['texts'] = batch_text
            inputs["offset_mapping"] = inputs["offset_mapping"].tolist()

            inputs = {**inputs, **label_inputs}
            inputs = self.build_batch_inputs(inputs)

            outputs = self.model(**inputs)
            output_list.extend(outputs['predictions'])

        if not return_dict:
            return output_list[0] if single_sentence else output_list
        else:
            return set2json(output_list[0]) if single_sentence else [set2json(o) for o in output_list]


class W2NERPredictor(PredictorBase):
    """ 基于w2ner的predictor
    """

    @torch.no_grad()
    def predict(
            self,
            text: Union[str, List[str]],
            batch_size: int = 8,
            max_length: int = 512,
            return_dict: bool = True,
    ) -> Union[dict, List[dict]]:

        single_sentence = False
        if isinstance(text, str):
            text = [text]
            single_sentence = True

        text = [t.replace(" ", "-") for t in text]  # 防止空格导致位置预测偏移

        output_list = []
        total_batch = len(text) // batch_size + (1 if len(text) % batch_size > 0 else 0)
        collate_fn = DataCollatorForW2Ner()
        for batch_id in tqdm(range(total_batch), desc="Predicting"):
            batch_text = text[batch_id * batch_size: (batch_id + 1) * batch_size]
            inputs = [self._process(example, max_length) for example in batch_text]

            inputs = collate_fn(inputs)
            inputs = self.build_batch_inputs(inputs)

            outputs = self.model(**inputs)
            output_list.extend(outputs['predictions'])

        if not return_dict:
            return output_list[0] if single_sentence else output_list
        else:
            return set2json(output_list[0]) if single_sentence else [set2json(o) for o in output_list]

    def _process(self, text, max_length):
        # sourcery skip: dict-comprehension, identity-comprehension, inline-immediately-returned-variable
        tokens = [self.tokenizer.tokenize(word) for word in text[:max_length - 2]]
        pieces = [piece for pieces in tokens for piece in pieces]
        _input_ids = self.tokenizer.convert_tokens_to_ids(pieces)
        _input_ids = np.array([self.tokenizer.cls_token_id] + _input_ids + [self.tokenizer.sep_token_id])

        length = len(tokens)
        _pieces2word = np.zeros((length, len(_input_ids)), dtype=np.bool)
        if self.tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)

        _dist_inputs = np.zeros((length, length), dtype=np.int)
        for k in range(length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k
        for i, j in itertools.product(range(length), range(length)):
            _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9 if _dist_inputs[i, j] < 0 else dis2idx[
                _dist_inputs[i, j]]

        _dist_inputs[_dist_inputs == 0] = 19

        _grid_mask = np.ones((length, length), dtype=np.bool)
        input_keys = ["input_ids", "pieces2word", "dist_inputs", "grid_mask"]

        encoded_inputs = {}
        for k, v in zip(input_keys, [_input_ids, _pieces2word, _dist_inputs, _grid_mask]):
            encoded_inputs[k] = list(v)

        encoded_inputs["text"] = text

        return encoded_inputs


PREDICTOR_MAP = {
    "mrc": PromptNERPredictor,
    "lear": LearNERPredictor,
    "w2ner": W2NERPredictor,
}


def get_auto_ner_predictor(model_name_or_path, model_name="crf", model_type="bert", schema2prompt=None, tokenizer=None,
                           device=None):
    predictor_class = PREDICTOR_MAP.get(model_name, NERPredictor)
    if tokenizer is None:
        tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path, do_lower_case=True)

    model = get_auto_ner_model(model_name=model_name, model_type=model_type)
    if model_name not in ["lear", "mrc"]:
        return predictor_class(model, model_name_or_path, tokenizer, device=device)
    assert schema2prompt is not None, "schema2prompt must be provided."
    return predictor_class(schema2prompt, model=model, model_name_or_path=model_name_or_path, tokenizer=tokenizer,
                           device=device)


class NERPipeline(object):
    def __init__(self, model_name_or_path, model_name="crf", model_type="bert", schema2prompt=None,
                 device=None, max_seq_len=512, batch_size=64, split_sentence=False, tokenizer=None):

        self._model_name_or_path = model_name_or_path
        self._model_name = model_name
        self._model_type = model_type
        self._schema2prompt = schema2prompt
        self._device = device
        self._max_seq_len = max_seq_len
        self._batch_size = batch_size
        self._split_sentence = split_sentence
        self._tokenizer = tokenizer

        self._prepare_predictor()

    def _prepare_predictor(self):
        logger.info(f">>> [Pytorch InferBackend of {self._model_type}-{self._model_name}] Creating Engine ...")
        self.inference_backend = get_auto_ner_predictor(self._model_name_or_path, self._model_name,
                                                        self._model_type, self._schema2prompt, self._tokenizer,
                                                        self._device)

    def __call__(self, inputs):

        texts = inputs
        if isinstance(texts, str):
            texts = [texts]

        max_prompt_len = len(max(self._schema2prompt.values())) if (self._schema2prompt is not None and self._model_name in["mrc", "lear"]) else 0
        max_predict_len = self._max_seq_len - max_prompt_len - 3

        short_input_texts, self.input_mapping = auto_splitter(texts, max_predict_len,
                                                              split_sentence=self._split_sentence)

        results = self.inference_backend.predict(short_input_texts, batch_size=self._batch_size,
                                                 max_length=self._max_seq_len, return_dict=False)
        results = self._auto_joiner(results, short_input_texts, self.input_mapping)
        return results

    def _auto_joiner(self, short_results, short_inputs, input_mapping):
        concat_results = []
        for k, vs in input_mapping.items():
            single_results = {}
            offset = 0
            for i, v in enumerate(vs):
                if i == 0:
                    single_results = short_results[v]
                else:
                    for res in short_results[v]:
                        tmp = res[0], res[1] + offset, res[2] + offset, res[3]
                        single_results.add(tmp)
                offset += len(short_inputs[v])
            single_results = set2json(single_results) if single_results else {}
            concat_results.append(single_results)
        return concat_results

    @property
    def seqlen(self):
        return self._max_seq_len

    @seqlen.setter
    def seqlen(self, value):
        self._max_seq_len = value

    @property
    def split(self):
        return self._split_sentence

    @split.setter
    def split(self, value):
        self._split_sentence = value


def vote(entities_list: List[dict], threshold=0.9) -> dict:
    """
    实体级别的投票方式
    entities_list: 所有模型预测出的一个文件的实体
    threshold: 大于阈值，模型预测出来的实体才能被选中
    """
    counts_dict = defaultdict(int)
    entities = defaultdict(list)

    for _entities in entities_list:
        for _type in _entities:
            for _ent in _entities[_type]:
                counts_dict[(_type, _ent["start"], _ent["end"], _ent["text"])] += 1

    for key in counts_dict:
        if counts_dict[key] >= (len(entities_list) * threshold):
            prob = counts_dict[key] / len(entities_list)
            dic = {"start": key[1], "end": key[2], "text": key[3], "probability": prob}
            entities[key[0]].append(dic)

    return entities


class EnsembleNERPredictor(object):
    """ 基于投票法预测实体
    """

    def __init__(self, predicators: List[NERPipeline]):
        self.predicators = predicators

    def predict(self, text: Union[str, List[str]], threshold=0.8) -> Union[dict, List[dict]]:
        single_sentence = False
        if isinstance(text, str):
            text = [text]
            single_sentence = True

        all_results = [predicator(text) for predicator in self.predicators]
        output_list = [vote(list(entities_list), threshold=threshold) for entities_list in zip(*all_results)]

        return output_list[0] if single_sentence else output_list
