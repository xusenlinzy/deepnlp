import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Optional
from ..utils import SequenceClassifierOutput, MODEL_MAP
from ...layers.dropouts import MultiSampleDropout


def get_auto_mdp_tc_model(model_type: str = "bert"):
    base_model, parent_model = MODEL_MAP[model_type]
    
    class SequenceClassifierWithMDP(parent_model):
        """
        1. 对每一层的[CLS]向量进行weight求和，以及添加Multi-Sample Dropout
        2. Multi-Sample Dropout可以理解为Dropout选择了输入集中的一个子集进行训练，相当于Stacking方法中的子模型
        Args:
            config: 模型的配置对象
        Reference:
            [1] Multi-Sample Dropout for Accelerated Training and Better Generalization
        """
        def __init__(self, config):
            config.output_hidden_states = True
            super().__init__(config)
            self.num_labels = config.num_labels
            self.bert = base_model(config)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

            n_weights = config.num_hidden_layers + 1
            weights_init = torch.zeros(n_weights).float()
            weights_init.data[:-1] = -3
            
            self.layer_weights = torch.nn.Parameter(weights_init)
            self.classifier = MultiSampleDropout(config.hidden_size, config.num_labels, 
                                K=getattr(config, 'k', 5), p=getattr(config, 'p', 0.5))
            
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

            outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            hidden_layers = outputs['hidden_states']

            cls_outputs = torch.stack([self.dropout(layer[:, 0, :]) for layer in hidden_layers], dim=2)
            cls_output = (torch.softmax(self.layer_weights, dim=0) * cls_outputs).sum(-1)
            logits = self.classifier(cls_output)

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
        
    return SequenceClassifierWithMDP
