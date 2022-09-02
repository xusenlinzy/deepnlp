<<<<<<< HEAD
# deepnlp
=======
# deepnlp

基于 `pytorch` 框架实现 `nlp` 各类任务的解决方案

# nlp任务

## 文本分类

**`STEP 1`:** 将训练数据转换为如下的 `json` 格式
```
{
  "text": "以色列大规模空袭开始！伊朗多个军事目标遭遇打击，誓言对等反击",
  "label": "news_military"
}
```

**`STEP 2`:** 选择文本分类模型，如 `BertForSequenceClassification`

```shell
export MODEL_DIR='hfl/chinese-roberta-wwm-ext'
export DATA_DIR=/home/xusenlin/nlp/dataset/tc/sentiment
export OUTPUR_DIR=/home/xusenlin/nlp/pytorch/outputs/tc/sentiment/bert/ 
export TASK_NAME=sentiment-bert

#-----------training-----------------
python task_sequence_classification.py \
  --model_type=bert \
  --pretrained_model_path=$MODEL_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --evaluate_during_training \
  --do_lower_case \
  --device_id='0' \
  --checkpoint_mode=max \
  --checkpoint_monitor=eval_f1 \
  --checkpoint_save_best \
  --data_dir=$DATA_DIR \
  --dataset=sentiment \
  --output_dir=$OUTPUR_DIR \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --pooler_type=cls \
  --learning_rate=2e-5 \
  --num_train_epochs=4 \
  --gradient_accumulation_steps=1 \
  --warmup_proportion=0.1 \
  --logging_steps=500 \
  --save_steps=500 \
  --seed=42
```


>>>>>>> 2c9ac12 (提交项目)
