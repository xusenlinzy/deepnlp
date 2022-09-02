import os
import sys

sys.path.append("..")

from transformers import BertConfig, BertTokenizerFast
from torchblocks.utils.options import Argparser
from torchblocks.utils.logger import Logger
from torchblocks.core import TextClassifierTrainer
from torchblocks.utils.device import prepare_device
from torchblocks.utils.seed import seed_everything
from torchblocks.tasks.tc import get_tc_train_dev_dataset, load_labels, DataCollatorWithPadding
from torchblocks.metrics.classification import F1Score, ClassReport
from torchblocks.tasks.tc import get_auto_tc_model


data_files = {"train": "train.json", "dev": "dev.json"}


def main():
    parser = Argparser.get_training_parser()
    group = parser.add_argument_group(title="tc", description="parameters for sequence classification task")
    group.add_argument('--model_name', default="fc", type=str, help='Model name for ner',
                       choices=['fc', 'mdp', 'rdrop'])
    
    group.add_argument("--alpha", default=4.0, type=float, help="alpha for r-drop")
    group.add_argument("--k", default=5, type=int, help="k for multi-sample dropout")
    group.add_argument("--p", default=0.5, type=float, help="p for multi-sample dropout")
    group.add_argument("--pooler_type", default='cls', type=str, help="pooler_type for sequence pooling")

    opts = parser.parse_args_from_parser(parser)

    logger = Logger(opts=opts)
    # device
    logger.info("initializing device")
    opts.device, opts.device_num = prepare_device(opts.device_id)
    seed_everything(opts.seed)
    
    model_class = get_auto_tc_model(model_name=opts.model_name, model_type=opts.model_type)

    # data processor
    logger.info("initializing data processor")
    tokenizer = BertTokenizerFast.from_pretrained(opts.pretrained_model_path, do_lower_case=opts.do_lower_case)

    label_path = os.path.join(opts.data_dir, 'label.txt')
    opts.label_list = load_labels(label_path)
    train_dataset, dev_dataset = get_tc_train_dev_dataset(opts.data_dir, data_files, tokenizer, opts.label_list,
                                                          task_name=opts.task_name,
                                                          train_max_seq_length=opts.train_max_seq_length,
                                                          eval_max_seq_length=opts.eval_max_seq_length)

    opts.num_labels = len(opts.label_list)

    # model
    logger.info("initializing model and config")
    config, unused_kwargs = BertConfig.from_pretrained(opts.pretrained_model_path,
                                                       return_unused_kwargs=True,
                                                       num_labels=opts.num_labels,)

    # FIXED: 默认`from_dict`中，只有config中有键才能设置值，这里强制设置
    for key, value in unused_kwargs.items(): setattr(config, key, value)
    model = model_class.from_pretrained(opts.pretrained_model_path, config=config)
    model.to(opts.device)

    # trainer
    logger.info("initializing traniner")
    metrics = [F1Score(task_type='multiclass', average='macro')]
    if opts.num_labels <= 30:
        metrics += [ClassReport(target_names=opts.label_list)]
    trainer = TextClassifierTrainer(opts=opts, model=model, metrics=metrics, logger=logger,
                                    collate_fn=DataCollatorWithPadding(tokenizer))

    if opts.do_train:
        trainer.train(train_data=train_dataset, dev_data=dev_dataset)


if __name__ == "__main__":
    main()
