import sys

sys.path.append("../..")

import os
from torchblocks.utils.options import Argparser, str2bool
from torchblocks.utils.logger import Logger
from torchblocks.metrics.sequence_labeling.ner_score import ExtractionScore
from torchblocks.core import ExtractionTrainer
from torchblocks.utils.device import prepare_device
from torchblocks.utils.seed import seed_everything
from torchblocks.tasks.ner import (
    get_auto_ner_model,
    get_auto_ner_collator,
    load_labels,
    get_ner_train_dev_dataset,
    get_mrc_ner_train_dev_dataset,
    get_w2ner_train_dev_dataset,
)
from transformers import BertConfig, BertTokenizerFast


QUERY = "找出下述句子中的"
ENTITY_2_QUERY = {
    "people-daily-ner-mrc": {
        "LOC": "按照地理位置划分的国家、城市、乡镇、大洲等",
        "PER": "人名和虚构的人物形象",
        "ORG": "组织包括公司、政府党派、学校、政府、新闻机构等"
    },
    "cluener-mrc": {
        "address": "按照地理位置划分的国家、城市、乡镇、大洲等",
        "book": "小说，杂志，习题集，教科书，教辅，食谱，书店里能买到的一类书籍，包含电子书",
        "company": "公司，集团，银行（央行，中国人民银行除外，二者属于政府机构）",
        "game": "常见的游戏，以及从小说，电视剧改编的游戏",
        "government": "包括中央行政机关和地方行政机关两级",
        "movie": "电影，也包括拍的一些在电影院上映的纪录片",
        "name": "人名和虚构的人物形象",
        "position": "古时候的职称：巡抚，知州，国师等。现代的总经理，记者，总裁，艺术家，收藏家等",
        "organization": "篮球队，足球队，社团等，小说里面的帮派如：少林寺，武当，峨眉等",
        "scene": "常见旅游景点如：长沙公园，深圳动物园，海洋馆，植物园，黄河，长江等"
    },
    "cmeee-mrc": {
        "dis": "疾病，主要包括疾病、中毒或受伤和器官或细胞受损",
        "sym": "临床表现，主要包括症状和体征",
        "pro": "医疗程序，主要包括检查程序、治疗或预防程序",
        "equ": "医疗设备，主要包括检查设备和治疗设备",
        "dru": "药物，是用以预防、治疗及诊断疾病的物质",
        "ite": "医学检验项目，是取自人体的材料进行血液学、细胞学等方面的检验",
        "bod": "身体，主要包括身体物质和身体部位",
        "dep": "部门科室，医院的各职能科室",
        "mic": "微生物类，一般是指细菌、病毒、真菌、支原体、衣原体、螺旋体等八类微生物"
    }
}

data_files = {"train": "train.json", "dev": "dev.json"}


def main():  # sourcery skip: low-code-quality, remove-redundant-if, split-or-ifs, swap-if-else-branches
    parser = Argparser.get_training_parser()
    group = parser.add_argument_group(title="ner", description="parameters for ner task")
    group.add_argument('--model_name', default="crf", type=str, help='Model name for ner',
                       choices=['crf', 'softmax', 'span', 'mrc', 'tplinker', 'global-pointer', 'lear', 'cascade-crf', 'w2ner'])

    # global-pointer
    group.add_argument('--head_size', default=64, type=int, help='The dim of Positional embedding')
    group.add_argument('--use_rope', type=str2bool, default="false", help='whether to use rotary position embeddings')
    group.add_argument('--is_sparse', type=str2bool, default="false", help='whether to use sparse labels')
    group.add_argument("--head_type", type=str, default="efficient_global_pointer",
                       choices=['global_pointer', 'efficient_global_pointer', 'biaffine', 'unlabeled_entity'])

    # tplinkerplus
    group.add_argument("--decode_thresh", type=float, default=0.0)
    group.add_argument("--shaking_type", type=str, default='cln', help='shaking_type for tplinkerplus')

    # span
    group.add_argument("--start_thresh", type=float, default=0.0)  # 0.5 for lear
    group.add_argument("--end_thresh", type=float, default=0.0)  # 0.5 for lear

    # lear
    group.add_argument("--nested", type=str2bool, default="false", help='whether to perform nested ner')

    # mrc
    group.add_argument("--use_label_embed", type=str2bool, default="false", help='whether to use label embed in mrc ner')

    # w2ner
    group.add_argument("--use_last_4_layers", type=str2bool, default="true", help='whether to use last 4 layers of bert')
    group.add_argument("--dist_emb_size", type=int, default=20)
    group.add_argument("--type_emb_size", type=int, default=20)
    group.add_argument("--lstm_hidden_size", type=int, default=512)
    group.add_argument("--conv_hidden_size", type=int, default=96)
    group.add_argument("--biaffine_size", type=int, default=512)
    group.add_argument("--ffn_hidden_size", type=int, default=288)

    # loss type
    group.add_argument("--loss_type", type=str, default="cross_entropy",
                       choices=['cross_entropy', 'focal_loss', 'label_smoothing_ce'])

    opts = parser.parse_args_from_parser(parser)
    logger = Logger(opts=opts)

    # device
    logger.info("initializing device")
    opts.device, opts.device_num = prepare_device(opts.device_id)
    seed_everything(opts.seed)
    
    model_class = get_auto_ner_model(model_name=opts.model_name, model_type=opts.model_type)
    data_collator = get_auto_ner_collator(model_name=opts.model_name)

    # data processor
    logger.info("initializing data processor")
    tokenizer = BertTokenizerFast.from_pretrained(opts.pretrained_model_path, do_lower_case=opts.do_lower_case)

    if opts.model_name not in ['mrc', 'lear']:
        label_path = os.path.join(opts.data_dir, 'label.txt')
        opts.label_list = load_labels(label_path)
    elif opts.model_name == 'mrc':
        opts.label_list = ENTITY_2_QUERY[opts.task_name]
    else:
        opts.label_list = ENTITY_2_QUERY[opts.task_name.replace("lear", "mrc")]

    dataset_fct = get_ner_train_dev_dataset
    if opts.model_name == 'w2ner':
        dataset_fct = get_w2ner_train_dev_dataset
    elif opts.model_name == 'mrc':
        dataset_fct = get_mrc_ner_train_dev_dataset
        
    train_dataset, dev_dataset = dataset_fct(
        opts.data_dir, data_files, tokenizer, opts.label_list,
        train_max_seq_length=opts.train_max_seq_length, eval_max_seq_length=opts.eval_max_seq_length
    )

    # model
    logger.info("initializing model and config")
    if opts.model_name == 'tplinker':
        num_labels = len(opts.label_list)
        label2id = {v: i for i, v in enumerate(opts.label_list)}
        id2label = {v: k for k, v in label2id.items()}

        config, unused_kwargs = BertConfig.from_pretrained(
            opts.pretrained_model_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            shaking_type=opts.shaking_type,
            decode_thresh=opts.decode_thresh,
            return_unused_kwargs=True,
        )
        collate_fn = data_collator(tokenizer, num_labels=num_labels)

    elif opts.model_name == 'global-pointer':
        num_labels = len(opts.label_list)
        label2id = {v: i for i, v in enumerate(opts.label_list)}
        id2label = {v: k for k, v in label2id.items()}

        config, unused_kwargs = BertConfig.from_pretrained(
            opts.pretrained_model_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            head_size=opts.head_size,
            use_rope=opts.use_rope,
            is_sparse=opts.is_sparse,
            head_type=opts.head_type,
            decode_thresh=opts.decode_thresh,
            return_unused_kwargs=True,
        )
        collate_fn = data_collator(tokenizer, num_labels=num_labels, is_sparse=opts.is_sparse)

    elif opts.model_name == 'crf':
        crf_labels = ['O'] + [f"B-{l}" for l in opts.label_list] + [f"I-{l}" for l in opts.label_list]
        num_labels = len(crf_labels)
        label2id = {v: i for i, v in enumerate(crf_labels)}
        id2label = {v: k for k, v in label2id.items()}

        config, unused_kwargs = BertConfig.from_pretrained(
            opts.pretrained_model_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            return_unused_kwargs=True,
        )
        collate_fn = data_collator(tokenizer, num_labels=len(opts.label_list))

    elif opts.model_name == 'cascade-crf':
        labels = ['O'] + opts.label_list
        num_labels = len(labels)
        label2id = {v: i for i, v in enumerate(labels)}
        id2label = {v: k for k, v in label2id.items()}

        config, unused_kwargs = BertConfig.from_pretrained(
            opts.pretrained_model_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            return_unused_kwargs=True,
        )
        collate_fn = data_collator(tokenizer)

    elif opts.model_name == 'span':
        span_labels = ['O'] + opts.label_list
        num_labels = len(span_labels)
        label2id = {v: i for i, v in enumerate(span_labels)}
        id2label = {v: k for k, v in label2id.items()}

        config, unused_kwargs = BertConfig.from_pretrained(
            opts.pretrained_model_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            start_thresh=opts.start_thresh,
            end_thresh=opts.end_thresh,
            loss_type=opts.loss_type,
            return_unused_kwargs=True,
        )
        collate_fn = data_collator(tokenizer)

    elif opts.model_name == 'mrc':
        config, unused_kwargs = BertConfig.from_pretrained(
            opts.pretrained_model_path,
            label_list=opts.label_list,
            num_labels=len(opts.label_list),
            use_label_embed=opts.use_label_embed,
            loss_type=opts.loss_type,
            return_unused_kwargs=True,
        )
        collate_fn = data_collator(tokenizer, num_labels=len(opts.label_list))

    elif opts.model_name == 'lear':
        num_labels = len(opts.label_list)
        label2id = {v: i for i, v in enumerate(opts.label_list.keys())}
        id2label = {v: k for k, v in label2id.items()}

        config, unused_kwargs = BertConfig.from_pretrained(
            opts.pretrained_model_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            nested=opts.nested,
            start_thresh=opts.start_thresh,
            end_thresh=opts.end_thresh,
            return_unused_kwargs=True,
        )
        collate_fn = data_collator(tokenizer, label_annotations=opts.label_list.values(), nested=opts.nested)

    elif opts.model_name == 'w2ner':
        w2ner_label_list = ["NONE", "NNW"] + opts.label_list
        num_labels = len(w2ner_label_list)
        label2id = {v: i for i, v in enumerate(w2ner_label_list)}
        id2label = {v: k for k, v in label2id.items()}

        config, unused_kwargs = BertConfig.from_pretrained(
            opts.pretrained_model_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            use_last_4_layers=opts.use_last_4_layers,
            dist_emb_size=opts.dist_emb_size,
            type_emb_size=opts.type_emb_size,
            lstm_hidden_size=opts.lstm_hidden_size,
            conv_hidden_size=opts.conv_hidden_size,
            biaffine_size=opts.biaffine_size,
            ffn_hidden_size=opts.ffn_hidden_size,
            return_unused_kwargs=True,
        )
        collate_fn = data_collator()

    # FIXED: 默认`from_dict`中，只有config中有键才能设置值，这里强制设置
    for key, value in unused_kwargs.items(): setattr(config, key, value)
    model = model_class.from_pretrained(opts.pretrained_model_path, config=config)
    model.to(opts.device)

    # trainer
    logger.info("initializing trainer")
    metrics = [ExtractionScore()]
    trainer = ExtractionTrainer(opts=opts, model=model, metrics=metrics, logger=logger, collate_fn=collate_fn,
                                wandb_config=unused_kwargs, project="NER")

    if opts.do_train:
        trainer.train(train_data=train_dataset, dev_data=dev_dataset)


if __name__ == "__main__":
    main()
