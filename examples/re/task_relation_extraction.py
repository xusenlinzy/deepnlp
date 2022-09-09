import sys

sys.path.append("../..")

import os
import itertools
from transformers import BertConfig, BertTokenizerFast
from torchblocks.utils.options import Argparser
from torchblocks.utils.logger import Logger
from torchblocks.metrics.sequence_labeling.ner_score import ExtractionScore
from torchblocks.core import ExtractionTrainer
from torchblocks.utils.device import prepare_device
from torchblocks.utils.seed import seed_everything
from torchblocks.tasks.ere import (
    load_labels,
    get_re_train_dev_dataset,
    get_auto_re_model,
    get_auto_re_collator,
)


data_files = {"train": "train.json", "dev": "dev.json"}


def main():  # sourcery skip: low-code-quality
    parser = Argparser.get_training_parser()
    group = parser.add_argument_group(title="re", description="parameters for re task")
    group.add_argument('--model_name', default="gplinker", type=str, help='Model name for re',
                       choices=['casrel', 'gplinker', 'tplinker', 'grte', 'spn', 'prgc', 'pfn'])
    # casrel
    group.add_argument("--start_thresh", type=float, default=0.5)
    group.add_argument("--end_thresh", type=float, default=0.5)
    
    # gplinker
    group.add_argument('--head_size', default=64, type=int, help='The dim of Positional embedding')
    
    # pfn network
    group.add_argument('--pfn_hidden_size', default=300, type=int, help='The dim of PFN hidden size')
    group.add_argument('--dropout', default=0.1, type=float, help='The dropout rate of PFN Network')
    
    # tplinkerplus
    group.add_argument("--decode_thresh", type=float, default=0.0)
    group.add_argument("--shaking_type", type=str, default='cln', help='Shaking type for tplinkerplus')
    
    # grte
    group.add_argument("--rounds", type=int, default=3, help="Number of rounds for GRTE")
    
    # prgc
    group.add_argument('--emb_fusion', type=str, default='concat', choices=['concat', 'sum'])
    group.add_argument('--corres_mode', type=str, default='biaffine')
    group.add_argument('--biaffine_hidden_size', type=int, default=128)
    group.add_argument('--negative_ratio', type=int, default=4, help="negtive samples for every positive sample")
    group.add_argument('--rel_threshold', type=float, default=0.1, help="threshold for relations decode")
    group.add_argument('--corres_threshold', type=float, default=0.5, help="threshold for corresponding decode")
    
    # spn
    group.add_argument('--num_generated_triples', type=int, default=10)
    group.add_argument('--num_decoder_layers', type=int, default=3)
    group.add_argument('--matcher', type=str, default="avg", choices=['avg', 'min'])
    group.add_argument('--na_rel_coef', type=float, default=1)
    group.add_argument('--rel_loss_weight', type=float, default=1)
    group.add_argument('--head_ent_loss_weight', type=float, default=2)
    group.add_argument('--tail_ent_loss_weight', type=float, default=2)
    group.add_argument('--max_span_length', type=int, default=10)
    group.add_argument('--n_best_size', type=int, default=20)

    opts = parser.parse_args_from_parser(parser)
    logger = Logger(opts=opts)

    # device
    logger.info("initializing device")
    opts.device, opts.device_num = prepare_device(opts.device_id)
    seed_everything(opts.seed)
    
    model_class = get_auto_re_model(model_name=opts.model_name, model_type=opts.model_type)
    data_collator = get_auto_re_collator(model_name=opts.model_name)
    
    # data processor
    logger.info("initializing data processor")
    tokenizer = BertTokenizerFast.from_pretrained(opts.pretrained_model_path, do_lower_case=opts.do_lower_case)

    label_path = os.path.join(opts.data_dir, 'label.txt')
    opts.label_list = load_labels(label_path)
    predicate2id = {t: idx for idx, t in enumerate(opts.label_list)}
    id2predicate = {idx: t for t, idx in predicate2id.items()}
    train_dataset, dev_dataset = get_re_train_dev_dataset(opts.data_dir, data_files, tokenizer, opts.label_list,
                                                          train_max_seq_length=opts.train_max_seq_length,
                                                          eval_max_seq_length=opts.eval_max_seq_length)

    # model
    logger.info("initializing model and config")
    if opts.model_name == 'casrel':
        config, unused_kwargs = BertConfig.from_pretrained(
            opts.pretrained_model_path, 
            return_unused_kwargs=True, 
            num_predicates=len(opts.label_list),
            predicate2id=predicate2id,
            id2predicate=id2predicate, 
            start_thresh=opts.start_thresh, 
            end_thresh=opts.end_thresh,
        )
        collate_fn = data_collator(tokenizer, num_predicates=len(opts.label_list))
        
    elif opts.model_name == 'tplinker':
        link_types = [
            "SH2OH",  # subject head to object head
            "OH2SH",  # object head to subject head
            "ST2OT",  # subject tail to object tail
            "OT2ST",  # object tail to subject tail
        ]
        tags = ["=".join([rel, lk]) for lk, rel in itertools.product(link_types, opts.label_list)]
        tags.append("DEFAULT=EH2ET")
        label2id = {t: idx for idx, t in enumerate(tags)}
        id2label = {idx: t for t, idx in label2id.items()}
        
        config, unused_kwargs = BertConfig.from_pretrained(
            opts.pretrained_model_path, 
            return_unused_kwargs=True, 
            num_predicates=len(opts.label_list),
            id2label=id2label, 
            label2id=label2id,
            shaking_type=opts.shaking_type, 
            decode_thresh=opts.decode_thresh, 
            predicate2id=predicate2id,
            id2predicate=id2predicate,
        )
        collate_fn = data_collator(tokenizer, num_predicates=len(opts.label_list))
        
    elif opts.model_name == 'gplinker':
        config, unused_kwargs = BertConfig.from_pretrained(
            opts.pretrained_model_path, 
            return_unused_kwargs=True, 
            num_predicates=len(opts.label_list),
            predicate2id=predicate2id,
            id2predicate=id2predicate, 
            decode_thresh=opts.decode_thresh, 
            head_size=opts.head_size,
        )
        collate_fn = data_collator(tokenizer, num_predicates=len(opts.label_list))
        
    elif opts.model_name == 'grte':
        tags = ["N/A", "SS", "MSH", "MST", "SMH", "SMT", "MMH", "MMT"]
        label2id = {t: idx for idx, t in enumerate(tags)}
        id2label = {idx: t for t, idx in label2id.items()}
        
        config, unused_kwargs = BertConfig.from_pretrained(
            opts.pretrained_model_path, 
            return_unused_kwargs=True, 
            num_labels=len(tags), 
            predicate2id=predicate2id,
            id2predicate=id2predicate, 
            id2label=id2label, 
            label2id=label2id, 
            num_predicates=len(opts.label_list),
            rounds=opts.rounds,
        )
        collate_fn = data_collator(tokenizer, num_predicates=len(opts.label_list), label2id=label2id)
        
    elif opts.model_name == 'spn':
        config, unused_kwargs = BertConfig.from_pretrained(
            opts.pretrained_model_path, 
            return_unused_kwargs=True, 
            num_predicates=len(opts.label_list),
            predicate2id=predicate2id,
            id2predicate=id2predicate, 
            num_decoder_layers=opts.num_decoder_layers,
            num_generated_triples=opts.num_generated_triples,
            matcher=opts.matcher, 
            na_rel_coef=opts.na_rel_coef, 
            rel_loss_weight=opts.rel_loss_weight,
            head_ent_loss_weight=opts.head_ent_loss_weight, 
            tail_ent_loss_weight=opts.tail_ent_loss_weight,
            max_span_length=opts.max_span_length, 
            n_best_size=opts.n_best_size,
        )
        collate_fn = data_collator(tokenizer)
        
    elif opts.model_name == 'prgc':
        tags = ["O", "B-ENT", "I-ENT"]
        label2id = {t: idx for idx, t in enumerate(tags)}
        id2label = {idx: t for t, idx in label2id.items()}
        
        config, unused_kwargs = BertConfig.from_pretrained(
            opts.pretrained_model_path, 
            return_unused_kwargs=True, 
            num_predicates=len(opts.label_list),
            id2label=id2label, 
            label2id=label2id,
            rel_threshold=opts.rel_threshold, 
            decode_thresh=opts.corres_threshold, 
            predicate2id=predicate2id,
            id2predicate=id2predicate,
            emb_fusion=opts.emb_fusion, 
            corres_mode=opts.corres_mode, 
            biaffine_hidden_size=opts.biaffine_hidden_size,
        )
        collate_fn = data_collator(tokenizer, num_predicates=len(opts.label_list), negative_ratio=opts.negative_ratio)
        
    elif opts.model_name == 'pfn':
        config, unused_kwargs = BertConfig.from_pretrained(
            opts.pretrained_model_path, 
            return_unused_kwargs=True, 
            num_predicates=len(opts.label_list),
            predicate2id=predicate2id,
            id2predicate=id2predicate,
            decode_thresh=opts.decode_thresh,
            pfn_hidden_size=opts.pfn_hidden_size,
            dropout=opts.dropout,
        )
        collate_fn = data_collator(tokenizer, num_predicates=len(opts.label_list))

    # FIXED: 默认`from_dict`中，只有config中有键才能设置值，这里强制设置
    for key, value in unused_kwargs.items(): setattr(config, key, value)
    model = model_class.from_pretrained(opts.pretrained_model_path, config=config)
    model.to(opts.device)

    # trainer
    logger.info("initializing traniner")
    metrics = [ExtractionScore()]
    trainer = ExtractionTrainer(opts=opts, model=model, tokenizer=tokenizer, metrics=metrics, logger=logger,
                                collate_fn=collate_fn, wandb_config=unused_kwargs, project="RE")

    if opts.do_train:
        trainer.train(train_data=train_dataset, dev_data=dev_dataset)


if __name__ == "__main__":
    main()
