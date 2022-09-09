import torch
import numpy as np
from tqdm import tqdm
from typing import List, Union
from transformers import BertTokenizerFast
from .auto import get_auto_tc_model
from ..predictor_base import PredictorBase
from ..uie.utils import logger, tqdm


class TextClassificationPredictor(PredictorBase):
    """
    A class for Text Classification task predictor.
    """
    @torch.no_grad()
    def predict(
            self,
            text_a: Union[str, List[str]],
            text_b: Union[str, List[str]] = None,
            batch_size: int = 64,
            max_length: int = 512,
    ) -> Union[dict, List[dict]]:
        # sourcery skip: boolean-if-exp-identity, remove-unnecessary-cast
        
        single_sentence = False
        if isinstance(text_a, str):
            text_a = [text_a]
            if text_b is not None and isinstance(text_b, str):
                text_b = [text_b]
            single_sentence = True

        output_list = []
        total_batch = len(text_a) // batch_size + (1 if len(text_a) % batch_size > 0 else 0)
        for batch_id in tqdm(range(total_batch), desc="Predicting"):
            batch_text_a = text_a[batch_id * batch_size: (batch_id + 1) * batch_size]
            if text_b is not None:
                batch_text_b = text_b[batch_id * batch_size: (batch_id + 1) * batch_size]
                inputs = self.tokenizer(
                    batch_text_a,
                    batch_text_b,
                    max_length=max_length,
                    padding=True,
                    truncation='only_second',
                    return_offsets_mapping=False,
                    return_tensors="pt",
                )
            else:
                inputs = self.tokenizer(
                    batch_text_a,
                    max_length=max_length,
                    truncation=True,
                    return_offsets_mapping=False,
                    padding=True,
                    return_tensors="pt",
                )
        
            inputs = self.build_batch_inputs(inputs)
            outputs = self.model(**inputs)
            outputs = np.asarray(outputs['logits']).argmax(-1)
            output_list.extend(outputs)

        return output_list[0] if single_sentence else output_list


def get_auto_tc_predictor(model_name_or_path, model_name="fc", model_type="bert", tokenizer=None, device=None):
    if tokenizer is None:
        tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path, do_lower_case=True)
    model = get_auto_tc_model(model_name=model_name, model_type=model_type)
    return TextClassificationPredictor(model, model_name_or_path, tokenizer=tokenizer, device=device)


class TextClassificationPipeline(object):
    def __init__(self, model_name_or_path, model_name="fc", model_type="bert", 
                 device=None, max_seq_len=512, batch_size=64, tokenizer=None):

        self._model_name_or_path = model_name_or_path
        self._model_name = model_name
        self._model_type = model_type
        self._device = device
        self._max_seq_len = max_seq_len
        self._batch_size = batch_size
        self._tokenizer = tokenizer

        self._prepare_predictor()

    def _prepare_predictor(self):
        logger.info(f">>> [Pytorch InferBackend of {self._model_type}-{self._model_name}] Creating Engine ...")
        self.inference_backend = get_auto_tc_predictor(self._model_name_or_path, self._model_name, 
                                                       self._model_type, self._tokenizer, self._device)

    def __call__(self, text_a, text_b=None):
        return self.inference_backend.predict(text_a, text_b, batch_size=self._batch_size, max_length=self._max_seq_len)

    @property
    def seqlen(self):
        return self._max_seq_len

    @seqlen.setter
    def seqlen(self, value):
        self._max_seq_len = value
