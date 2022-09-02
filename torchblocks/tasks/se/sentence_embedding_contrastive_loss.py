import torch
from typing import Optional
from ..utils import SentenceEmbeddingOutput
from ...layers.pooling import Pooler
from ...losses.contrastive_loss import ContrastiveLoss
from transformers import BertPreTrainedModel, BertModel


class BertForContrastiveLoss(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)

        self.pooler_type = config.pooler_type if hasattr(config, 'pooler_type') else 'cls'
        self.pooling = Pooler(self.pooler_type)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
    ) -> SentenceEmbeddingOutput:
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            output_hidden_states=True)

        pooled_output = self.pooling(outputs, attention_mask)
        pooled_output_1 = pooled_output[::2, :]  # 第一个句子在奇数行
        pooled_output_2 = pooled_output[1::2, :]  # 第二个句子在偶数行

        cos_dist = 1 - torch.cosine_similarity(pooled_output_1, pooled_output_2)
        loss = self.compute_loss([cos_dist, labels]) if labels is not None else None
        return SentenceEmbeddingOutput(loss=loss, embeddings=pooled_output)

    def compute_loss(self, inputs):
        dist, labels = inputs[:2]
        # 1. 取出真实的标签（偶数行为重复的标签）
        labels = labels[::2].flatten()  # tensor([1, 0, 1]) 真实的标签
        loss_fct = ContrastiveLoss()
        return loss_fct(dist, labels)
