通过运行以下命令进行模型微调：

```bash
cd examples/tc
bash run.sh
```

`run.sh` 主要配置参数说明：

- `data_dir`：数据集所在文件夹，包含两个文件：`train.json` 和 `dev.json`
- `output_dir`：模型输出路径
- `model_type`：模型类型，目前支持 `bert`、`roformer`、`ernie` 系列模型
- `model_name`：模型名称，目前支持 `fc`、`mdp`、`rdrop`
- `pretrained_model_path`：预训练模型路径，与 [`transformers`](https://github.com/huggingface/transformers) 中提供的模型名称和路径一致。
- `task_name`：任务名称，可自定义，用于打印训练日志
- `do_train`：是否进行训练，使用字符串 `"true"` 表示执行
- `evaluate_during_training`：训练时对验证集进行评估，使用字符串 `"true"` 表示执行
- `do_lower_case`：将字符串转小写，使用字符串 `"true"` 表示执行
- `device_id`：`gpu`设备的序号
- `checkpoint_monitor`：验证集评估指标名称
- `checkpoint_mode`：验证集评估指标模式，`max` 表示越大越好，`min` 表示越小越好
- `checkpoint_save_best`：保存最优模型，使用字符串 `"true"` 表示执行
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