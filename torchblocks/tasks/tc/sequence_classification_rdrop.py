import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Optional
from ..utils import SequenceClassifierOutput, MODEL_MAP


def get_auto_rdrop_tc_model(model_type: str = "bert"):
    base_model, parent_model = MODEL_MAP[model_type]
    
    class BertWithRDrop(parent_model):
        """
        + 📖 通过增加一个正则项，来强化模型对`Dropout`的鲁棒性
        + 📖 使得不同的`Dropout`下模型的输出基本一致，因此能降低这种不一致性
        + 📖 促进“模型平均”与“权重平均”的相似性，使得简单关闭`Dropout`的效果等价于多`Dropout`模型融合的结果
        
        Args:
            `config`: 模型的配置对象
            
        Reference:
            ⭐️ [R-Drop: Regularized Dropout for Neural Networks.](https://spaces.ac.cn/archives/8496) 
            🚀 [Official Code](https://github.com/dropreg/R-Drop)
        """
        def __init__(self, config):
            super(BertWithRDrop, self).__init__(config)
            self.num_labels = config.num_labels
            self.bert = base_model(config)
            config.hidden_dropout_prob = 0.3
            self.config = config
            
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
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

            outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            pooled_output = outputs['pooler_output']
            logits = self.dropout(self.classifier(pooled_output))

            if self.training:
                outputs2 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                pooled_output2 = outputs2['pooler_output']
                logits2 = self.dropout(self.classifier(pooled_output2))

            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                alpha = getattr(self.config, "alpha", 4.0)
                loss = alpha * loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                if self.training:
                    loss += alpha * loss_fct(logits2.view(-1, self.num_labels), labels.view(-1))

                    p = torch.log_softmax(logits[0].view(-1, self.num_labels), dim=-1)
                    p_tec = torch.softmax(logits[0].view(-1, self.num_labels), dim=-1)
                    q = torch.log_softmax(logits2[-1].view(-1, self.num_labels), dim=-1)
                    q_tec = torch.softmax(logits2[-1].view(-1, self.num_labels), dim=-1)

                    kl_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none').sum()
                    reverse_kl_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none').sum()
                    loss += 0.5 * (kl_loss + reverse_kl_loss) / 2

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
            
    return BertWithRDrop
