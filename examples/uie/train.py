import sys

sys.path.append("../..")

import os
import torch
from torchblocks.tasks.uie import UIE
from torchblocks.tasks.uie.utils import IEDataset, SpanEvaluator
from transformers import BertTokenizerFast

from torchblocks.utils.options import Argparser
from torchblocks.utils.logger import Logger
from torchblocks.core import TrainerBase
from torchblocks.callback import ProgressBar
from torchblocks.utils.device import prepare_device
from torchblocks.utils.seed import seed_everything

MODEL_CLASSES = {
    'bert': (BertTokenizerFast, UIE),
}


def collate_fn(features):
    batch = {"input_ids": [x[0] for x in features], "token_type_ids": [x[1] for x in features],
             "attention_mask": [x[2] for x in features], "start_positions": [x[3] for x in features],
             "end_positions": [x[4] for x in features]}
    return {k: torch.tensor(v) for k, v in batch.items()}


class UIETrainer(TrainerBase):
    def evaluate(self, dev_data, prefix_metric=None):
        """
        Evaluate the model on a validation set
        """
        eval_dataloader = self.build_eval_dataloader(dev_data)
        self.build_record_tracker()
        self.reset_metrics()

        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        for step, batch in enumerate(eval_dataloader):
            batch = self.predict_forward(batch)
            if 'loss' in batch and batch['loss'] is not None:
                self.records['loss_meter'].update(batch['loss'], n=1)

            start_prob, end_prob = batch['start_prob'], batch['end_prob']
            start_ids, end_ids = batch['start_positions'], batch['end_positions']

            num_correct, num_infer, num_label = self.metrics[0].compute(start_prob, end_prob, start_ids, end_ids)
            self.metrics[0].update(num_correct, num_infer, num_label)
            pbar.step(step)

        self.records['result']['eval_loss'] = self.records['loss_meter'].avg
        self.update_metrics(prefix_metric)
        self.print_evaluate_result()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def update_metrics(self, prefix):
        precision, recall, f1 = self.metrics[0].accumulate()
        value = {"precision": precision, "recall": recall, "f1": f1}
        if not isinstance(value, dict):
            raise ValueError("metric value type: expected one of (float, dict,None)")
        prefix = '' if prefix is None else f"{prefix}_"
        self.records['result'].update({f"{prefix}eval_{k}": v for k, v in value.items()})


if __name__ == '__main__':
    parser = Argparser.get_training_parser()
    parser.add_argument("-t", "--train_file", default=None, required=True,
                        type=str, help="The file_name of train set.")
    parser.add_argument("-d", "--dev_file", default=None, required=True,
                        type=str, help="The file_name of dev set.")
    opts = parser.parse_args_from_parser(parser)
    logger = Logger(opts=opts)

    # device
    logger.info("initializing device")
    opts.device, opts.device_num = prepare_device(opts.device_id)
    seed_everything(opts.seed)
    tokenizer_class, model_class = MODEL_CLASSES[opts.model_type]

    # data processor
    logger.info("initializing data processor")
    tokenizer = tokenizer_class.from_pretrained(opts.pretrained_model_path, do_lower_case=opts.do_lower_case)

    train_path = os.path.join(opts.data_dir, opts.train_input_file)
    dev_path = os.path.join(opts.data_dir, opts.eval_input_file)
    train_ds = IEDataset(train_path, tokenizer=tokenizer, max_seq_len=opts.train_max_seq_length)
    dev_ds = IEDataset(dev_path, tokenizer=tokenizer, max_seq_len=opts.eval_max_seq_length)

    model = model_class.from_pretrained(opts.pretrained_model_path)
    model.to(opts.device)

    # trainer
    logger.info("initializing trainer")
    metrics = [SpanEvaluator()]
    trainer = UIETrainer(opts=opts, model=model, metrics=metrics, logger=logger, collate_fn=collate_fn, project="UIE")

    if opts.do_train:
        trainer.train(train_data=train_ds, dev_data=dev_ds)

    tokenizer.save_pretrained("./checkpoint/uie_bert_v0/checkpoint-eval_f1-best")
