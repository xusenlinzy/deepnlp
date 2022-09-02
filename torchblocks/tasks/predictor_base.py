import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel


class PredictorBase(object):
    """
    A class for base predictor.
    """
    
    keys_to_ignore_on_gpu = ['offset_mapping', 'texts', 'target']  # batch不存放在gpu中的变量

    def __init__(
            self,
            model: PreTrainedModel,
            model_name_or_path: str,
            tokenizer: PreTrainedTokenizerBase,
            device: str = None,
    ):
        self.tokenizer = tokenizer
        self.model = model.from_pretrained(model_name_or_path)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = self.model.to(self.device)
        self.model.eval()

    def build_batch_inputs(self, batch):
        """
        Sent all model inputs to the appropriate device (GPU on CPU)
        return:
         The inputs are in a dictionary format
        """
        return {
            key: (value.to(self.device) if ((key not in self.keys_to_ignore_on_gpu) and (value is not None)) else value)
            for key, value in batch.items()}

    @torch.no_grad()
    def predict(self, text, **kwargs):
        raise NotImplementedError('Method [predict] should be implemented.')
    