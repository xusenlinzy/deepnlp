import torch
import torch.nn as nn
from typing import Optional, List, Any
from ..utils import SequenceLabelingOutput, MODEL_MAP
from ...layers.crf import CRF
from ...utils.tensor import sequence_padding
from ...metrics.sequence_labeling.scheme import get_entities


def get_extended_attention_mask(attention_mask: torch.Tensor):
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    return (1.0 - extended_attention_mask) * -10000.0


def get_auto_crf_ner_model(model_type: str = "bert"):
    base_model, parent_model = MODEL_MAP[model_type]

    class CRFForNer(parent_model):
        """
        基于`BERT`的`CRF`实体识别模型
        + 📖 `BERT`编码器提取`token`的语义特征
        + 📖 `CRF`层学习标签之间的约束关系

        Args:
            `config`: 模型的配置对象
        """

        def __init__(self, config):
            super().__init__(config)
            self.bert = base_model(config)
            self.config = config
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            self.crf = CRF(num_tags=config.num_labels, batch_first=True)
            
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
            sequence_output = self.dropout(outputs[0])
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
            decode_ids = self.crf.decode(logits, mask).squeeze(0)  # (batch_size, seq_length)
            decode_labels = []
            for text, ids, mask, mapping in zip(texts, decode_ids, mask, offset_mapping):
                decode_label = [self.config.id2label[id.item()] for id, m in zip(ids, mask) if m > 0][:-1]  # [CLS], [SEP]
                decode_label = get_entities(decode_label)
                decode_label = [(l[0], mapping[l[1]][0].item(), mapping[l[2]][1].item(), text[mapping[l[1]][0]: mapping[l[2]][1]]) for l in decode_label]
                decode_labels.append(set(decode_label))
            return decode_labels

        def compute_loss(self, inputs):
            logits, labels, mask = inputs[:3]
            return -1 * self.crf(emissions=logits, tags=labels, mask=mask)

    return CRFForNer


def get_auto_cascade_crf_ner_model(model_type: str = "bert"):
    base_model, parent_model = MODEL_MAP[model_type]

    class CascadeCRFForNer(parent_model):
        """
        基于`BERT`的层级`CRF`实体识别模型
        + 📖 `BERT`编码器提取`token`的语义特征
        + 📖 第一阶段`CRF`层学习`BIO`标签之间的约束关系抽取所有实体
        + 📖 第二阶段采用一个线性层对实体进行分类
        
        Args:
            `config`: 模型的配置对象
        """

        def __init__(self, config):
            super().__init__(config)
            self.bert = base_model(config)
            self.config = config
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

            self.dense1 = nn.Linear(config.hidden_size, 3)
            self.crf = CRF(num_tags=3, batch_first=True)
            self.dense2 = nn.Linear(config.hidden_size, config.num_labels)

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

        def get_entity_logits(self, sequence_output, entity_ids):
            btz, entity_count, _ = entity_ids.shape
            entity_ids = entity_ids.reshape(btz, -1, 1).repeat(1, 1, self.config.hidden_size)
            entity_states = torch.gather(sequence_output, dim=1, index=entity_ids).reshape(btz, entity_count, -1,
                                                                                           self.config.hidden_size)
            entity_states = torch.mean(entity_states, dim=2)  # 取实体首尾hidden_states的均值
            return self.dense2(entity_states)  # [btz, 实体个数，实体类型数]

        def forward(
                self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                entity_ids: Optional[torch.Tensor] = None,
                entity_labels: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                texts: Optional[List[str]] = None,
                offset_mapping: Optional[List[Any]] = None,
                target: Optional[List[Any]] = None,
                return_decoded_labels: Optional[bool] = True,
        ) -> SequenceLabelingOutput:

            outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            sequence_output = self.dropout(outputs[0])
            logits = self.dense1(sequence_output)

            if entity_ids is not None:
                entity_logits = self.get_entity_logits(sequence_output, entity_ids)

            loss, predictions = None, None
            if labels is not None and entity_ids is not None and entity_labels is not None:
                loss = self.compute_loss([logits, entity_logits, entity_labels, labels, attention_mask])

            if not self.training and return_decoded_labels:  # 训练时无需解码
                predictions = self.decode(sequence_output, logits, attention_mask, texts, offset_mapping)

            return SequenceLabelingOutput(
                loss=loss,
                logits=logits,
                predictions=predictions,
                groundtruths=target,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        def decode(self, sequence_output, logits, mask, texts, offset_mapping):
            decode_ids = self.crf.decode(logits, mask).squeeze(0)  # (batch_size, seq_length)
            entity_ids = []
            BIO_MAP = getattr(self.config, 'BIO_MAP', {0: "O", 1: "B-ENT", 2: "I-ENT"})
            for ids, mask in zip(decode_ids, mask):
                decode_label = [BIO_MAP[id.item()] for id, m in zip(ids, mask) if m > 0][:-1]  # [CLS], [SEP]
                decode_label = get_entities(decode_label)
                if len(decode_label) > 0:
                    entity_ids.append([[l[1], l[2]] for l in decode_label])
                else:
                    entity_ids.append([[0, 0]])

            entity_ids = torch.from_numpy(sequence_padding(entity_ids)).to(sequence_output.device)
            entity_logits = self.get_entity_logits(sequence_output, entity_ids)
            entity_preds = torch.argmax(entity_logits, dim=-1)  # [btz, 实体个数]

            decode_labels = []
            for i, (entities, text, mapping) in enumerate(zip(entity_ids, texts, offset_mapping)):
                tmp = set()
                for j, ent in enumerate(entities):
                    s, e, p = ent[0].item(), ent[1].item(), entity_preds[i][j].item()
                    if s * e * p != 0:
                        _start, _end = mapping[s][0].item(), mapping[e][1].item()
                        tmp.add((
                            self.config.id2label[p], _start, _end, text[_start: _end]
                        ))
                decode_labels.append(tmp)
            return decode_labels

        def compute_loss(self, inputs):
            logits, entity_logits, entity_labels, labels, mask = inputs[:5]
            loss = -1 * self.crf(emissions=logits, tags=entity_labels, mask=mask)
            loss += 4 * nn.CrossEntropyLoss(ignore_index=0)(entity_logits.view(-1, self.config.num_labels),
                                                            labels.flatten())
            return loss

    return CascadeCRFForNer
