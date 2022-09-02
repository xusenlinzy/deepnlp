import importlib.util
from sklearn.model_selection import KFold


def is_apex_available():
    return importlib.util.find_spec("apex") is not None


def k_fold_train(examples, trainer, opts, model, metrics, collate_fn, logger, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=opts.seed)
    for fold, (train_idx, dev_idx) in enumerate(kf.split(examples)):
        logger.info(f'Start to train the {fold} fold')
        train_dataset = examples.iloc[train_idx]
        dev_dataset = examples.iloc[dev_idx]

        # trainer
        logger.info("initializing trainer of {fold}th fold")
        trainer = trainer(opts=opts, model=model, metrics=metrics, logger=logger, collate_fn=collate_fn)
        trainer.opts.output_dir += f"FOLD-{fold}/"
        trainer.train(train_data=train_dataset, dev_data=dev_dataset)
