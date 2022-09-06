import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Optional, List, Any
from ..utils import SpanOutput, MODEL_MAP
from ...losses.span_loss import SpanLoss


def get_auto_span_ner_model(model_type: str = "bert"):
    base_model, parent_model = MODEL_MAP[model_type]

    class SpanForNer(parent_model):
        """
        基于`BERT`的`Span`实体识别模型
        1. 对于每个`token`分别进行对应实体类型的起始位置判断
        2. 分类数目为实体类型数目+1（非实体）
        
        Args:
            `config`: 模型的配置对象
        """

        def __init__(self, config):
            super().__init__(config)
            self.bert = base_model(config)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

            self.start_fc = nn.Linear(config.hidden_size, config.num_labels)
            self.end_fc = nn.Linear(config.hidden_size, config.num_labels)
            self.loss_type = getattr(config, 'loss_type', 'cross_entropy')
            
            if hasattr(config, 'use_task_id') and config.use_task_id and model_type == "ernie":
                # Add task type embedding to BERT
                task_type_embeddings = nn.Embedding(config.task_type_vocab_size, config.hidden_size)
                self.bert.embeddings.task_type_embeddings = task_type_embeddings

                def hook(module, input, output):
                    return output + task_type_embeddings(
                        torch.zeros(input[0].size(), dtype=torch.int64, device=input[0].device))

                self.bert.embeddings.word_embeddings.register_forward_hook(hook)

            # Initialize weights and apply final processing
            self.post_init()

        def forward(
                self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                start_positions: Optional[torch.Tensor] = None,
                end_positions: Optional[torch.Tensor] = None,
                texts: Optional[List[str]] = None,
                offset_mapping: Optional[List[Any]] = None,
                target: Optional[List[Any]] = None,
                return_decoded_labels: Optional[bool] = True,
        ) -> SpanOutput:

            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            sequence_output = outputs[0]
            sequence_output = self.dropout(sequence_output)

            start_logits = self.start_fc(sequence_output)
            end_logits = self.end_fc(sequence_output)

            loss, predictions = None, None
            if start_positions is not None and end_positions is not None:
                loss = self.compute_loss([
                    start_logits, end_logits, start_positions, end_positions, attention_mask
                ])

            if not self.training and return_decoded_labels:  # 训练时无需解码
                predictions = self.decode(
                    start_logits, end_logits, attention_mask, texts, offset_mapping,
                    start_thresh=getattr(self.config, "start_thresh", 0.0),
                    end_thresh=getattr(self.config, "end_thresh", 0.0),
                )

            return SpanOutput(
                loss=loss,
                start_logits=start_logits,
                end_logits=end_logits,
                predictions=predictions,
                groundtruths=target,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        def decode(self, start_logits, end_logits, sequence_mask, texts, offset_mapping, 
                   start_thresh=0.0, end_thresh=0.0, **kwargs):
            other_id = self.config.label2id["O"]
            id2label = self.config.id2label
            max_span_length = kwargs.get("max_span_length", 10)

            # # TODO: 概率先判断是否为实体
            # start_probs = start_logits.softmax(dim=-1)  # (batch_size, seqlen, num_labels)
            # other_probs = start_probs[..., other_id]  # (batch_size, seqlen)
            # other_probs = torch.where(other_probs < start_thresh, torch.zeros_like(other_probs), other_probs)
            # start_probs[..., other_id] = other_probs
            # start_probs, start_labels = start_probs.max(dim=-1)

            # end_probs = end_logits.softmax(dim=-1)  # (batch_size, seqlen, num_labels)
            # other_probs = end_probs[..., other_id]  # (batch_size, seqlen)
            # other_probs = torch.where(other_probs < end_thresh, torch.zeros_like(other_probs), other_probs)
            # end_probs[..., other_id] = other_probs
            # end_probs, end_labels = end_probs.max(dim=-1)

            start_labels, end_labels = torch.argmax(start_logits, -1), torch.argmax(end_logits, -1)

            decode_labels = []
            batch_size = sequence_mask.size(0)
            for i in range(batch_size):
                decode_label = set()
                label_start_map = defaultdict(list)  # 每种类别设置起始标志，处理实体重叠情况，如：
                # start: [0, 0, 1, 0, 2, 0, 0, 0]
                # end:   [0, 0, 0, 0, 0, 2, 1, 0]
                for pos, (s, e, m) in enumerate(zip(start_labels[i], end_labels[i], sequence_mask[i])):
                    s, e, m = s.item(), e.item(), m.item()
                    if m == 0: break
                    if s != other_id:
                        label_start_map[s].append(pos)  # 以下两个功能：
                        # 1. 进入s类型span，以label_start_map[s]标记;
                        # 2. 若在s类型span内，但重新遇到s类型span起始时，追加位置
                    if e != other_id:  # 在e类型span内（已包括处理单个token的实体）
                        for start in label_start_map[e]:
                            start, end, label = start, pos, id2label[e]  # [CLS]
                            if end - start < max_span_length:
                                text, mapping = texts[i], offset_mapping[i]
                                _start, _end = mapping[start][0], mapping[end][1]
                                decode_label.add((label, _start, _end, text[_start: _end]))  # 遇到结束位置，新建span
                        label_start_map[e] = []
                decode_labels.append(decode_label)
            return decode_labels

        def compute_loss(self, inputs):
            start_logits, end_logits, start_positions, end_positions, masks = inputs[:5]
            loss_fct = SpanLoss(self.config.num_labels, loss_type=self.loss_type)
            return loss_fct(preds=(start_logits, end_logits), target=(start_positions, end_positions), masks=masks)

    return SpanForNer
