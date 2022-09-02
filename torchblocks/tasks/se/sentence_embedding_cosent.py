import torch
from typing import Optional
from ..utils import SentenceEmbeddingOutput
from ...layers.pooling import Pooler
from transformers import BertPreTrainedModel, BertModel


class BertForCoSent(BertPreTrainedModel):
    """
    优化cos值的新方案——CoSENT：比Sentence-BERT更有效的句向量方案
    https://spaces.ac.cn/archives/8847
    """

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
        loss = None if labels is None else self.compute_loss([pooled_output, labels])
        return SentenceEmbeddingOutput(loss=loss, embeddings=pooled_output)

    def compute_loss(self, inputs):
        preds, labels = inputs[:2]
        # 1. 取出真实的标签（偶数行为重复的标签）
        labels = labels[::2]  # tensor([1, 0, 1]) 真实的标签
        # 2. 对输出的句子向量进行l2归一化   后面只需要对应为相乘  就可以得到cos值了
        norms = (preds ** 2).sum(axis=1, keepdims=True) ** 0.5
        # preds = preds / torch.clip(norms, 1e-8, torch.inf)
        preds = preds / norms

        # 3. 奇偶向量相乘
        preds = torch.sum(preds[::2] * preds[1::2], dim=1) * 20
        # 4. 取出负例-正例的差值
        preds = preds[:, None] - preds[None, :]  # 这里是算出所有位置 两两之间余弦的差值
        # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
        labels = labels[:, None] < labels[None, :]  # 取出负例-正例的差值
        labels = labels.float()
        preds = preds - (1 - labels) * 1e12
        preds = preds.view(-1)
        preds = torch.cat((torch.tensor([0.0], device=preds.device), preds), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
        return torch.logsumexp(preds, dim=0)
