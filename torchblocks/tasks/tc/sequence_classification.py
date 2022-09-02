import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from typing import Optional
from ..utils import SequenceClassifierOutput, MODEL_MAP
from ...layers.pooling import Pooler


def get_auto_fc_tc_model(model_type: str = "bert"):
    base_model, parent_model = MODEL_MAP[model_type]
    
    class SequenceClassification(parent_model):
        """
        基于BERT的文本分类模型
        Args:
            config: 模型的配置对象
        """
        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.config = config
            self.bert = base_model(config)
            
            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            
            self.dropout = nn.Dropout(classifier_dropout)
            self.pooling = Pooler(getattr(config, 'pooler_type', 'cls'))
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            
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
        ) -> SequenceClassifierOutput:

            outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
            pooled_output = self.dropout(self.pooling(outputs, attention_mask))
            logits = self.classifier(pooled_output)

            loss = self.compute_loss([logits, labels]) if labels is not None else None
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        def compute_loss(self, inputs):
            logits, labels = inputs[:2]
            loss_fct = CrossEntropyLoss()
            return loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

    return SequenceClassification
