import torch
from tqdm import tqdm
from collections import defaultdict
from typing import List, Union
from transformers import BertTokenizerFast
from .auto import get_auto_re_model
from ..predictor_base import PredictorBase
from ...utils.common import auto_splitter
from ..uie.utils import logger, tqdm


def set2json(labels) -> dict:
    """ 将三元组集合根据关系类型转换为字典    
    """
    res = {}
    for _sub, _rel, _obj in labels:
        dic = {"subject": _sub, "object": _obj}
        if _rel not in res:
            res[_rel] = [dic]
        else:
            res[_rel].append(dic)
    return res


class REPredictor(PredictorBase):
    """
    A class for RE task predictor.
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
        
        if not return_dict:
            return output_list[0] if single_sentence else output_list
        else:
            return set2json(output_list[0]) if single_sentence else [set2json(o) for o in output_list]


def get_auto_re_predictor(model_name_or_path, model_name="gplinker", model_type="bert", tokenizer=None, device=None):
    if tokenizer is None:
        tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path, do_lower_case=True)
    model = get_auto_re_model(model_name=model_name, model_type=model_type)
    return REPredictor(model, model_name_or_path, tokenizer=tokenizer, device=device)


class REPipeline(object):
    def __init__(self, model_name_or_path, model_name="gplinker", model_type="bert", device=None, 
                 max_seq_len=512, batch_size=64, split_sentence=False, tokenizer=None):

        self._model_name_or_path = model_name_or_path
        self._model_name = model_name
        self._model_type = model_type
        self._device = device
        self._max_seq_len = max_seq_len
        self._batch_size = batch_size
        self._split_sentence = split_sentence
        self._tokenizer = tokenizer

        self._prepare_predictor()

    def _prepare_predictor(self):
        logger.info(f">>> [Pytorch InferBackend of {self._model_type}-{self._model_name}] Creating Engine ...")
        self.inference_backend = get_auto_re_predictor(self._model_name_or_path, self._model_name, 
                                                       self._model_type, self._tokenizer, self._device)

    def __call__(self, inputs):

        texts = inputs
        if isinstance(texts, str):
            texts = [texts]

        max_predict_len = self._max_seq_len - 2
        short_input_texts, self.input_mapping = auto_splitter(texts, max_predict_len, split_sentence=self._split_sentence)

        results = self.inference_backend.predict(short_input_texts, batch_size=self._batch_size,
                                                 max_length=self._max_seq_len, return_dict=False)
        results = self._auto_joiner(results, self.input_mapping)
        return results

    def _auto_joiner(self, short_results, input_mapping):
        concat_results = []
        for k, vs in input_mapping.items():
            group_results = [short_results[v] for v in vs if len(short_results[v]) > 0]
            single_results = set2json(set.union(*group_results))
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


def vote(triples_list: List[dict], threshold=0.9) -> dict:
    """
    三元组级别的投票方式
    entities_list: 所有模型预测出的一个文件的实体
    threshold: 大于阈值，模型预测出来的实体才能被选中
    """
    counts_dict = defaultdict(int)
    triples = defaultdict(list)

    for _triples in triples_list:
        for _rel in _triples:
            for _triple in _triples[_rel]:
                counts_dict[(_rel, _triple["subject"], _triple["object"])] += 1

    for key in counts_dict:
        if counts_dict[key] >= (len(triples_list) * threshold):
            prob = counts_dict[key] / len(triples_list)
            dic = {"subject": key[1], "object": key[2], "probability": prob}
            triples[key[0]].append(dic)

    return triples


class EnsembleREPredictor(object):
    """ 基于投票法预测三元组
    """
    def __init__(self, predicators: List[REPipeline]):
        self.predicators = predicators
        
    def predict(self, text: Union[str, List[str]], threshold=0.8) -> Union[dict, List[dict]]:
        single_sentence = False
        if isinstance(text, str):
            text = [text]
            single_sentence = True
        
        all_results = [predicator(text) for predicator in self.predicators]
        output_list = [vote(list(spo_list), threshold=threshold) for spo_list in zip(*all_results)]

        return output_list[0] if single_sentence else output_list
    