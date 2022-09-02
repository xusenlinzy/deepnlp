import torch
import itertools
import torch.nn as nn
from typing import Optional, List, Any
from ..utils import RelationExtractionOutput, MODEL_MAP
from ...layers.global_pointer import HandshakingKernel
from ...losses.cross_entropy import MultiLabelCategoricalCrossEntropy


def get_auto_tplinker_re_model(model_type: str = "bert"):
    base_model, parent_model = MODEL_MAP[model_type]

    class TPLinkerPlus(parent_model):
        """
        基于`BERT`的`TPLinker`关系抽取模型
        + 📖 模型的整体思路将三元组抽取问题转化为`token`对之间的链接问题
        + 📖 对于每一个关系类型，主体-客体的链接关系为：首首、尾尾以及实体首尾
        + 📖 对于`token`对采用矩阵上三角展开的方式进行多标签分类
        
        Args:
            `config`: 模型的配置对象
        
        Reference:
            ⭐️ [TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking.](https://aclanthology.org/2020.coling-main.138.pdf)
            🚀 [Official Code](https://github.com/131250208/TPlinker-joint-extraction)
        """

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.bert = base_model(config)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

            self.handshaking_kernel = HandshakingKernel(config.hidden_size, config.shaking_type)
            self.out_dense = nn.Linear(config.hidden_size, config.num_predicates * 4 + 1)
            
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
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                texts: Optional[List[str]] = None,
                offset_mapping: Optional[List[Any]] = None,
                target: Optional[List[Any]] = None,
        ) -> RelationExtractionOutput:

            outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            sequence_output = self.dropout(outputs.last_hidden_state)  # [B, L, H]

            # shaking_hiddens: (batch_size, shaking_seq_len, hidden_size)
            shaking_hiddens = self.handshaking_kernel(sequence_output)
            # shaking_logits: (batch_size, shaking_seq_len, tag_size)
            shaking_logits = self.out_dense(shaking_hiddens)

            loss, predictions = None, None
            if labels is not None:
                loss = self.compute_loss([shaking_logits, labels])

            if not self.training:
                seq_len = input_ids.shape[1]
                predictions = self.decode(shaking_logits, seq_len, texts, offset_mapping)

            return RelationExtractionOutput(
                loss=loss,
                logits=None,
                predictions=predictions,
                groundtruths=target,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        def decode(self, shaking_logits, seq_len, texts, offset_mapping):
            # sourcery skip: low-code-quality
            all_spo_list = []
            batch_size = shaking_logits.size(0)
            for bs in range(batch_size):
                text, mapping = texts[bs], offset_mapping[bs]
                shaking_idx2matrix_idx = [(s, e) for s in range(seq_len) for e in list(range(seq_len))[s:]]
                head_ind2entities = {}
                spoes = set()

                _shaking_logits = shaking_logits[bs].cpu()
                matrix_spots = self.get_spots_fr_shaking_tag(shaking_idx2matrix_idx, _shaking_logits)

                for sp in matrix_spots:
                    tag = self.config.id2label[sp[2]]
                    ent_type, link_type = tag.split("=")
                    # for an entity, the start position can not be larger than the end pos.
                    if link_type != "EH2ET" or sp[0] > sp[1]:
                        continue

                    entity = {
                        "type": ent_type,
                        "tok_span": [sp[0], sp[1]],
                    }
                    # take ent_head_pos as the key to entity list
                    head_key = sp[0]
                    if head_key not in head_ind2entities:
                        head_ind2entities[head_key] = []
                    head_ind2entities[head_key].append(entity)

                # tail link
                tail_link_memory_set = set()
                for sp in matrix_spots:
                    tag = self.config.id2label[sp[2]]
                    rel, link_type = tag.split("=")

                    if link_type == "ST2OT":
                        rel = self.config.predicate2id[rel]
                        tail_link_memory = (rel, sp[0], sp[1])
                        tail_link_memory_set.add(tail_link_memory)
                    elif link_type == "OT2ST":
                        rel = self.config.predicate2id[rel]
                        tail_link_memory = (rel, sp[1], sp[0])
                        tail_link_memory_set.add(tail_link_memory)

                # head link
                for sp in matrix_spots:
                    tag = self.config.id2label[sp[2]]
                    rel, link_type = tag.split("=")

                    if link_type == "SH2OH":
                        rel = self.config.predicate2id[rel]
                        subj_head_key, obj_head_key = sp[0], sp[1]
                    elif link_type == "OH2SH":
                        rel = self.config.predicate2id[rel]
                        subj_head_key, obj_head_key = sp[1], sp[0]
                    else:
                        continue

                    if (
                            subj_head_key not in head_ind2entities
                            or obj_head_key not in head_ind2entities
                    ):
                        # no entity start with subj_head_key and obj_head_key
                        continue

                    # all entities start with this subject head
                    subj_list = head_ind2entities[subj_head_key]
                    # all entities start with this object head
                    obj_list = head_ind2entities[obj_head_key]

                    for subj, obj in itertools.product(subj_list, obj_list):
                        tail_link_memory = (rel, subj["tok_span"][1], obj["tok_span"][1])

                        if tail_link_memory not in tail_link_memory_set:
                            # no such relation
                            continue
                        spoes.add(
                            (
                                text[
                                mapping[subj["tok_span"][0]][0]: mapping[
                                    subj["tok_span"][1]
                                ][1]
                                ],
                                self.config.id2predicate[rel],
                                text[
                                mapping[obj["tok_span"][0]][0]: mapping[
                                    obj["tok_span"][1]
                                ][1]
                                ],
                            )
                        )
                all_spo_list.append(set(spoes))
            return all_spo_list

        def get_spots_fr_shaking_tag(self, shaking_idx2matrix_idx, shaking_outputs):
            """
            shaking_tag -> spots
            shaking_tag: (shaking_seq_len, tag_id)
            spots: [(start, end, tag), ]
            """
            spots = []
            decode_thresh = getattr(self.config, "decode_thresh", 0.0)
            pred_shaking_tag = (shaking_outputs > decode_thresh).long()
            nonzero_points = torch.nonzero(pred_shaking_tag, as_tuple=False)
            for point in nonzero_points:
                shaking_idx, tag_idx = point[0].item(), point[1].item()
                pos1, pos2 = shaking_idx2matrix_idx[shaking_idx]
                spot = (pos1, pos2, tag_idx)
                spots.append(spot)
            return spots

        def compute_loss(self, inputs):
            shaking_logits, labels = inputs[:2]
            loss_fct = MultiLabelCategoricalCrossEntropy()
            return loss_fct(shaking_logits, labels)

    return TPLinkerPlus
