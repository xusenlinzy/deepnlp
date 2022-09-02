import torch
from typing import Optional
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from ..utils import SiameseClassificationOutput


class BertForSiameseClassification(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForSiameseClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.seq_relationship = nn.Linear(config.hidden_size * 3, config.num_labels)
        self.init_weights()

    def forward(
        self, 
        input_ids_a: Optional[torch.Tensor] = None,
        input_ids_b: Optional[torch.Tensor] = None, 
        attention_mask_a: Optional[torch.Tensor] = None,
        attention_mask_b: Optional[torch.Tensor] = None,
        token_type_ids_a: Optional[torch.Tensor] = None,
        token_type_ids_b: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> SiameseClassificationOutput:

        outputs_a = self.bert(input_ids_a, token_type_ids=token_type_ids_a, attention_mask=attention_mask_a)
        outputs_b = self.bert(input_ids_b, token_type_ids=token_type_ids_b, attention_mask=attention_mask_b)

        pooled_output = torch.cat([outputs_a[1], outputs_b[1], torch.abs(outputs_a[1] - outputs_b[1])], dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.seq_relationship(pooled_output)

        loss = self.compute_loss([logits, labels]) if labels is not None else None
        return SiameseClassificationOutput(
            loss=loss,
            logits=logits,
        )

    def compute_loss(self, inputs):
        logits, labels = inputs[:2]
        loss_fct = CrossEntropyLoss()
        return loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        