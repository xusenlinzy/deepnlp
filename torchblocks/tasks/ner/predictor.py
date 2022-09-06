import torch
import logging
import itertools
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import List, Union
from .processor import dis2idx, DataCollatorForW2Ner
from ..predictor_base import PredictorBase


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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
    ) -> Union[dict, List[dict]]:
        # sourcery skip: boolean-if-exp-identity, remove-unnecessary-cast
        single_sentence = False
        if isinstance(text, str):
            text = [text]
            single_sentence = True
        
        text = [t.replace(" ", "-") for t in text]  # 防止空格导致位置预测偏移

        output_list = []
        total_batch = len(text) // batch_size + (1 if len(text) % batch_size > 0 else 0)
        for batch_id in tqdm(range(total_batch)):
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

            inputs = self.build_batch_inputs(inputs)
            outputs = self.model(**inputs)
            output_list.extend(outputs['predictions'])
        
        return set2json(output_list[0]) if single_sentence else [set2json(o) for o in output_list]


class PromptNERPredictor(PredictorBase):
    """ 基于prompt的predictor
    """
    def __init__(self, schema2prompt: dict = None, **kwargs):
        self.schema2prompt = schema2prompt
        super().__init__(**kwargs)
        self.model.config.label_list = schema2prompt

    @torch.no_grad()
    def predict(self, text: Union[str, List[str]], max_length: int = 512):
        if isinstance(text, str):
            return self.single_sample_predict(text, max_length)
        else:
            return [self.single_sample_predict(sent, max_length) for sent in text]
    
    def single_sample_predict(self, text: str, max_length: int = 512):
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
        inputs = self.build_batch_inputs(inputs)
        outputs = self.model(**inputs)['predictions']
        
        return set2json(outputs[0])


class LearNERPredictor(PredictorBase):
    """ 基于LEAR的predictor
    """
    def __init__(self, schema2prompt: dict = None, **kwargs):
        self.schema2prompt = schema2prompt
        super().__init__(**kwargs)

    @torch.no_grad()
    def predict(self, text: Union[str, List[str]], max_length: int = 512, batch_size: int = 8) -> Union[dict, List[dict]]:
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
        for batch_id in tqdm(range(total_batch)):
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
            inputs = {**inputs, **label_inputs}
            inputs = self.build_batch_inputs(inputs)
            
            outputs = self.model(**inputs)
            output_list.extend(outputs['predictions'])
        
        return set2json(output_list[0]) if single_sentence else [set2json(o) for o in output_list]
    

class W2NERPredictor(PredictorBase):
    """ 基于w2ner的predictor
    """
    @torch.no_grad()
    def predict(self, text: Union[str, List[str]], max_length: int = 512, batch_size: int = 8) -> Union[dict, List[dict]]:
        single_sentence = False
        if isinstance(text, str):
            text = [text]
            single_sentence = True
            
        text = [t.replace(" ", "-") for t in text]  # 防止空格导致位置预测偏移
        
        output_list = []
        total_batch = len(text) // batch_size + (1 if len(text) % batch_size > 0 else 0)
        collate_fn = DataCollatorForW2Ner()
        for batch_id in tqdm(range(total_batch)):
            batch_text = text[batch_id * batch_size: (batch_id + 1) * batch_size]
            inputs = [self.process(example, max_length) for example in batch_text]
            
            inputs = collate_fn(inputs)
            inputs = self.build_batch_inputs(inputs)
            
            outputs = self.model(**inputs)
            output_list.extend(outputs['predictions'])
        
        return set2json(output_list[0]) if single_sentence else [set2json(o) for o in output_list]
    
    def process(self, text, max_length):
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
            _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9 if _dist_inputs[i, j] < 0 else dis2idx[_dist_inputs[i, j]]

        _dist_inputs[_dist_inputs == 0] = 19
        
        _grid_mask = np.ones((length, length), dtype=np.bool)
        input_keys = ["input_ids", "pieces2word", "dist_inputs", "grid_mask"]
        
        encoded_inputs = {}
        for k, v in zip(input_keys, [_input_ids, _pieces2word, _dist_inputs, _grid_mask]):
            encoded_inputs[k] = list(v)
            
        encoded_inputs["text"] = text
        
        return encoded_inputs


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
    def __init__(self, predicators: List[PredictorBase]):
        self.predicators = predicators
        
    def predict(self, text: Union[str, List[str]], max_length: int = 512, threshold=0.8) -> Union[dict, List[dict]]:
        single_sentence = False
        if isinstance(text, str):
            text = [text]
            single_sentence = True
        
        output_list = []
        for sent in text:
            entities_list = [predicator.predict(sent, max_length=max_length) for predicator in self.predicators]
            output_list.append(vote(entities_list, threshold=threshold))
            
        return output_list[0] if single_sentence else output_list
    