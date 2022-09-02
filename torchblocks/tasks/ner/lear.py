import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Any
from ..utils import SpanOutput, MODEL_MAP
from ...losses.span_loss import SpanLossForMultiLabel
from ...layers.lear import LabelFusionForToken, Classifier, MLPForMultiLabel


def get_auto_lear_ner_model(model_type: str = "bert"):
    base_model, parent_model = MODEL_MAP[model_type]

    class LEARForNer(parent_model):
        """
        基于`BERT`的`LEAR`实体识别模型
        + 📖 模型的整体思路将实体识别问题转化为每个实体类型下的`span`预测问题
        + 📖 模型的输入分为两个部分：原始的待抽取文本和所有标签对应的文本描述（先验知识）
        + 📖 原始文本和标签描述文本共享`BERT`的编码器权重
        + 📖 采用注意力机制融合标签信息到`token`特征中去
        
        Args:
            `config`: 模型的配置对象
        
        Reference:
            ⭐️ [Enhanced Language Representation with Label Knowledge for Span Extraction.](https://aclanthology.org/2021.emnlp-main.379.pdf)
            🚀 [Code](https://github.com/Akeepers/LEAR)
        """

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.bert = base_model(config)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

            # 将标签信息融合到token的特征当中
            self.label_fusion_layer = LabelFusionForToken(config.hidden_size)
            self.start_classifier = Classifier(config.hidden_size, config.num_labels)
            self.end_classifier = Classifier(config.hidden_size, config.num_labels)

            # 嵌套NER则增加一个span matrix的预测
            if config.nested:
                self.span_layer = MLPForMultiLabel(config.hidden_size * 2, config.num_labels)
                
            if hasattr(config, 'use_task_id') and config.use_task_id and model_type == "ernie":
                # Add task type embedding to BERT
                task_type_embeddings = nn.Embedding(config.task_type_vocab_size, config.hidden_size)
                self.bert.embeddings.task_type_embeddings = task_type_embeddings

                def hook(module, input, output):
                    return output + task_type_embeddings(
                        torch.zeros(input[0].size(), dtype=torch.int64, device=input[0].device))

                self.bert.embeddings.word_embeddings.register_forward_hook(hook)

        def forward(
                self,
                input_ids: Optional[torch.Tensor] = None,
                label_input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                label_attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                label_token_type_ids: Optional[torch.Tensor] = None,
                start_labels: Optional[torch.Tensor] = None,
                end_labels: Optional[torch.Tensor] = None,
                span_labels: Optional[torch.Tensor] = None,
                texts: Optional[List[str]] = None,
                offset_mapping: Optional[List[Any]] = None,
                target: Optional[List[Any]] = None,
                return_decoded_labels: Optional[bool] = True,
        ) -> SpanOutput:

            token_features = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
            token_features = self.dropout(token_features)

            label_features = \
                self.bert(label_input_ids, attention_mask=label_attention_mask, token_type_ids=label_token_type_ids)[0]
            fused_features = self.label_fusion_layer(token_features, label_features, label_attention_mask)

            start_logits = self.start_classifier(fused_features)
            end_logits = self.end_classifier(fused_features)

            span_logits = None
            if self.config.nested:
                seqlen = input_ids.shape[1]
                start_extend = fused_features.unsqueeze(2).expand(-1, -1, seqlen, -1, -1)
                end_extend = fused_features.unsqueeze(1).expand(-1, seqlen, -1, -1, -1)
                span_matrix = torch.cat((start_extend, end_extend), dim=-1)
                span_logits = self.span_layer(span_matrix)

            loss, predictions = None, None
            if start_labels is not None and end_labels is not None:
                if self.config.nested:
                    loss = self.compute_loss([
                        start_logits, end_logits, span_logits, start_labels, end_labels, span_labels, attention_mask
                    ])
                else:
                    loss = self.compute_loss([
                        start_logits, end_logits, start_labels, end_labels, attention_mask
                    ])

            if not self.training and return_decoded_labels:  # 训练时无需解码
                predictions = self.decode(start_logits, end_logits, span_logits, attention_mask, texts, offset_mapping)

            return SpanOutput(
                loss=loss,
                start_logits=start_logits,
                end_logits=end_logits,
                span_logits=span_logits,
                predictions=predictions,
                groundtruths=target)

        def decode(self, start_logits, end_logits, span_logits, attention_mask, texts, offset_mapping):
            # sourcery skip: low-code-quality
            bs, seqlen, num_labels = start_logits.shape
            decode_labels = []

            if not self.config.nested:
                seqlens = attention_mask.sum(1).cpu().numpy()
                start_preds, end_preds = torch.sigmoid(start_logits), torch.sigmoid(end_logits)
                start_preds, end_preds = start_preds.cpu().numpy(), end_preds.cpu().numpy()

                start_thresh = getattr(self.config, "start_thresh", 0.5)
                end_thresh = getattr(self.config, "end_thresh", 0.5)

                for i in range(bs):
                    text, mapping = texts[i], offset_mapping[i]
                    decode_label = set()
                    starts, ends = np.where(start_preds[i] > start_thresh), np.where(end_preds[i] > end_thresh)
                    for _start, c1 in zip(*starts):
                        if _start == 0 or _start >= seqlens[i] - 1:
                            continue
                        for _end, c2 in zip(*ends):
                            if _start <= _end < seqlens[i] - 1 and c1 == c2:
                                s, e = mapping[_start][0].item(), mapping[_end][1].item()
                                decode_label.add((
                                    self.config.id2label[c1], s, e, text[s: e]
                                ))
                                break  # 就近原则
                    decode_labels.append(decode_label)
                return decode_labels

            masks = attention_mask.unsqueeze(-1).expand(-1, -1, num_labels)

            start_label_masks = masks.unsqueeze(-2).expand(-1, -1, seqlen, -1).bool()
            end_label_masks = masks.unsqueeze(-3).expand(-1, seqlen, -1, -1).bool()

            # [batch_size, seq_len, num_labels]
            start_preds, end_preds = (start_logits > 0).bool, (end_logits > 0).bool()
            # [batch_size, seq_len, seq_len, num_labels]
            match_preds = span_logits > 0

            match_preds = (
                    match_preds & start_preds.unsqueeze(2).expand(-1, -1, seqlen, -1)
                    & end_preds.unsqueeze(1).expand(-1, seqlen, -1, -1)
            )

            match_label_masks = torch.triu((start_label_masks & end_label_masks, 0).permute(0, 3, 1, 2), 0).permute(0,
                                                                                                                    2,
                                                                                                                    3,
                                                                                                                    1)
            match_preds = match_preds & match_label_masks

            for i in range(bs):
                decode_label = set()
                preds = torch.where(match_preds[i] == True)
                for start_idx, end_idx, label_id in preds:
                    start_idx, end_idx, label_id = start_idx.item(), end_idx.item(), label_id.item()
                    _start, _end = mapping[start_idx][0].item(), mapping[end_idx][1].item()
                    decode_label.add((
                        self.config.id2label[label_id], _start, _end, text[_start: _end]
                    ))
            return decode_labels

        def compute_loss(self, inputs):
            loss_fct = SpanLossForMultiLabel()

            if not self.config.nested:
                start_logits, end_logits, start_labels, end_labels, mask = inputs[:5]
                loss = loss_fct(
                    (start_logits, end_logits),
                    (start_labels, end_labels), mask
                )
                return loss
            start_logits, end_logits, span_logits, start_labels, end_labels, span_labels, mask = inputs[:7]
            loss = loss_fct(
                (start_logits, end_logits, span_logits),
                (start_labels, end_labels, span_labels), mask, nested=True
            )
            return loss

    return LEARForNer
