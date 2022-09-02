import torch
import torch.nn as nn
from typing import Optional, List, Any
from ..utils import SequenceLabelingOutput, MODEL_MAP
from ...losses.focal_loss import FocalLoss
from ...losses.label_smoothing import LabelSmoothingCE
from ...metrics.sequence_labeling.scheme import get_entities


def get_auto_softmax_ner_model(model_type: str = "bert"):
    base_model, parent_model = MODEL_MAP[model_type]

    class SoftmaxForNer(parent_model):
        """
        基于`BERT`的`Softmax`实体识别模型
        
        Args:
            `config`: 模型的配置对象
        """

        def __init__(self, config):
            super().__init__(config)
            self.bert = base_model(config)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            loss_type = getattr(config, 'loss_type', 'cross_entropy')

            loss_fcts = {
                'cross_entropy': nn.CrossEntropyLoss(ignore_index=0),
                'focal_loss': FocalLoss(config.num_labels),
                'label_smoothing_ce': LabelSmoothingCE(ignore_index=0)
            }
            self.loss_fct = loss_fcts.get(loss_type, 'cross_entropy')
            
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
                labels: Optional[torch.Tensor] = None,
                texts: Optional[List[str]] = None,
                offset_mapping: Optional[List[Any]] = None,
                target: Optional[List[Any]] = None,
                return_decoded_labels: Optional[bool] = True,
        ) -> SequenceLabelingOutput:

            outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            sequence_output = outputs[0]
            sequence_output = self.dropout(sequence_output)
            logits = self.classifier(sequence_output)

            loss, predictions = None, None
            if labels is not None:
                loss = self.compute_loss([logits, labels, attention_mask])

            if not self.training and return_decoded_labels:  # 训练时无需解码
                predictions = self.decode(logits, attention_mask, texts, offset_mapping)

            return SequenceLabelingOutput(
                loss=loss,
                logits=logits,
                predictions=predictions,
                groundtruths=target,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        def decode(self, logits, mask, texts, offset_mapping):
            decode_ids = torch.argmax(logits, -1)  # (batch_size, seq_length)
            decode_labels = []
            for text, ids, mask, mapping in zip(texts, decode_ids, mask, offset_mapping):
                decode_label = [self.config.id2label[id.item()] for id, m in zip(ids, mask) if m > 0][
                               1:-1]  # [CLS], [SEP]
                decode_label = get_entities(decode_label)
                decode_label = [(l[0], mapping[l[1]][0].item(), mapping[l[2]][1].item(), text[mapping[l[1]][0]: mapping[l[2]][1]]) for l in decode_label]
                decode_labels.append(set(decode_label))
            return decode_labels

        def compute_loss(self, inputs):
            logits, labels, attention_mask = inputs[:3]
            active_loss = attention_mask.view(-1) == 1

            active_logits = logits.view(-1, self.config.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            return self.loss_fct(active_logits, active_labels)

    return SoftmaxForNer
