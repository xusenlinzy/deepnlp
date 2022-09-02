import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Union
from ..predictor_base import PredictorBase


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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
        for batch_id in tqdm(range(total_batch)):
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
    