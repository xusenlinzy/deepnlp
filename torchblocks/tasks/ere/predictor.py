import torch
import logging
from tqdm import tqdm
from collections import defaultdict
from typing import List, Union
from ..predictor_base import PredictorBase


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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
    