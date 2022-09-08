# deepnlp

基于 `pytorch` 框架实现 `nlp` 各类任务的解决方案

# nlp任务

## 1. 文本分类

**`STEP 1`**：将训练数据转换为如下的 `json` 格式
```json
{
  "text": "以色列大规模空袭开始！伊朗多个军事目标遭遇打击，誓言对等反击",
  "label": "news_military"
}
```

> 训练数据包含两个文件：`train.json` 和 `dev.json`。

**`STEP 2`**：执行训练脚本

支持的模型

  - ✅ `fc`：`BERT` 后面接一个全连接层
  - ✅ `mdp`：`BERT` 后面接一个层 [`MultiSampleDropout`](https://arxiv.org/abs/1905.09788)
  - ✅ `rdrop`：`BERT` 后面接一个全连接层并使用 [`R-Drop`](https://github.com/dropreg/R-Drop) 正则化

通过运行以下命令进行模型微调：

```commandline
cd examples/tc
```

```shell
python task_sequence_classification.py \
    --data_dir "/home/xusenlin/nlp/deepnlp/dataset/tc" \
    --output_dir "/home/xusenlin/nlp/deepnlp/examples/tc/outputs/mdp/" \
    --model_type "bert" \
    --model_name "mdp" \
    --pretrained_model_path "hfl/chinese-roberta-wwm-ext" \
    --task_name "mdp" \
    --do_train "true" \
    --evaluate_during_training "true" \
    --do_lower_case "true" \
    --device_id "0" \
    --checkpoint_mode "max" \
    --checkpoint_monitor "eval_f1" \
    --checkpoint_save_best "true" \
    --train_max_seq_length 64 \
    --eval_max_seq_length 64 \
    --per_gpu_train_batch_size 32 \
    --per_gpu_eval_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 10 \
    --gradient_accumulation_steps 1 \
    --warmup_proportion 0.1 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --seed 2022
```

可配置参数说明：

- `data_dir`：数据集所在文件夹，包含两个文件：`train.json` 和 `dev.json`
- `output_dir`：模型输出路径
- `model_type`：模型类型，目前支持 `bert`、`roformer`、`ernie` 系列模型
- `model_name`：模型名称，目前支持 `fc`、`mdp`、`rdrop`
- `pretrained_model_path`：预训练模型路径，与 [`transformers`](https://github.com/huggingface/transformers) 中提供的模型名称和路径一致。
- `task_name`：任务名称，可自定义，用于打印训练日志
- `do_train`：是否进行训练，使用字符串 "true"表示执行
- `evaluate_during_training`：训练时对验证集进行评估，使用字符串 "true"表示执行
- `do_lower_case`：将字符串转小写，使用字符串 "true"表示执行
- `device_id`：`gpu`设备的序号
- `checkpoint_monitor`：验证集评估指标名称
- `checkpoint_mode`：验证集评估指标模式，`max` 表示越大越好，`min` 表示越小越好
- `checkpoint_save_best`：保存最优模型，使用字符串 "true"表示执行
- `train_max_seq_length`: 训练集文本最大长度
- `eval_max_seq_length`：验证集文本最大长度
- `per_gpu_train_batch_size`：训练集批量大小
- `per_gpu_eval_batch_size`：验证集批量大小
- `learning_rate`：学习率大小
- `num_train_epochs`：训练轮次
- `gradient_accumulation_steps`：梯度累计的步数
- `warmup_proportion`：学习率预热的步数或者比例
- `logging_steps`：日志打印的间隔步数
- `save_steps`：模型保存的间隔步数
- `seed`: 随机种子，便于结果复现

