import torch
from dataclasses import dataclass
from typing import Optional, Union
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


###################################### Question Answering ##############################################

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        labels = ([feature.pop("label") for feature in features] if "label" in features[0].keys() else None)
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in
                              features]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # unflatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        if labels is None:  # for test
            return batch
        # add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
