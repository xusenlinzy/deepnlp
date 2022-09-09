import os
import sys
import math
import torch
import warnings
import wandb
import pandas as pd
import torch.nn as nn
from argparse import Namespace
from packaging import version

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers import AdamW, Adafactor

from .utils import is_apex_available
from ..utils.logger import Logger
from ..utils.common import check_object_type
from ..utils.seed import seed_everything
from ..utils.meter import AverageMeter
from ..utils.paths import save_pickle, json_to_text, load_model, is_file
from ..callback.file_writer import FileWriter
from ..callback import ModelCheckpoint, EarlyStopping, ProgressBar, EMA
from ..callback.adversarial import FGM, PGD, AWP, FreeLB, ALUM
from ..optims.lr_scheduler import get_lr_scheduler
from ..callback.model_checkpoint import (WEIGHTS_NAME,
                                         TRAINER_STATE_NAME,
                                         OPTIMIZER_NAME,
                                         SCHEDULER_NAME,
                                         SCALER_NAME)

if not sys.warnoptions:
    warnings.simplefilter("ignore")

_is_native_amp_available = False
if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


class TrainerBase:
    """Base class for iterative trainer."""
    keys_to_ignore_on_gpu = []  # batch不存放在gpu中的变量，比如'input_length’
    keys_to_ignore_on_result_save = ['input_ids', 'token_type_ids']  # eval和predict结果不存储的变量
    keys_to_ignore_on_checkpoint_save = []  # checkpoint中不存储的模块，比如optimizer'

    def __init__(self,
                 opts,
                 model,
                 tokenizer,
                 metrics,
                 logger,
                 optimizer=None,
                 scheduler=None,
                 adv_model=None,
                 model_checkpoint=None,
                 early_stopping=None,
                 collate_fn=None,
                 wandb_config=None,
                 project=None,
                 **kwargs):
        self.opts = opts
        self.model = model
        self.tokenizer = tokenizer
        self.metrics = metrics
        self.logger = logger
        self.scheduler = scheduler
        self.collate_fn = collate_fn

        self.global_step = 0
        self.device_num = getattr(opts, 'device_num', 0)
        self.warmup_steps = getattr(opts, 'warmup_steps', 0)
        self.num_train_epochs = getattr(opts, "num_train_epochs", 3)
        self.device = getattr(opts, 'device', torch.device("cpu"))
        self.max_grad_norm = getattr(opts, 'max_grad_norm', 0.0)
        self.warmup_proportion = getattr(opts, 'warmup_proportion', 0.1)
        self.gradient_accumulation_steps = getattr(opts, "gradient_accumulation_steps", 1)
        self.prefix = "_".join([opts.model_type, opts.task_name, opts.experiment_code])

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.build_writer()
        self.build_mixed_precision()

        if not isinstance(self.metrics, list):
            self.metrics = [self.metrics]

        check_object_type(object=self.metrics, check_type=list, name='metric')
        check_object_type(object=self.model, check_type=nn.Module, name='model')
        check_object_type(object=self.opts, check_type=Namespace, name='self.opts')
        check_object_type(object=self.logger, check_type=Logger, name='self.logger')

        # EMA
        if opts.ema_enable:
            self.logger.info('Using EMA')
            self.model_ema = EMA(model=self.model,
                                 decay=opts.ema_decay,
                                 device='cpu' if opts.model_ema_force_cpu else None)

        # Adversarial training
        if opts.adv_enable:
            msg = f"Using Adversarial training and type: {opts.adv_type}"
            self.logger.info(msg)
            self.adv_model = adv_model
            if adv_model is None:
                self.adv_model = self.build_adv_model()

        # optimizer
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = self.build_optimizer(model)

        # checkpoint
        self.model_checkpoint = model_checkpoint
        if model_checkpoint is None:
            self.model_checkpoint = ModelCheckpoint(
                mode=opts.checkpoint_mode,
                monitor=opts.checkpoint_monitor,
                ckpt_dir=opts.output_dir,
                verbose=opts.checkpoint_verbose,
                save_best=opts.checkpoint_save_best,
                keys_to_ignore_on_save=self.keys_to_ignore_on_checkpoint_save
            )

        # earlystopping
        self.early_stopping = early_stopping
        if early_stopping is None and opts.earlystopping_patience > 0:
            self.early_stopping = EarlyStopping(
                mode=opts.earlystopping_mode,
                patience=opts.earlystopping_patience,
                monitor=opts.earlystopping_monitor,
                save_state_path=opts.earlystopping_save_state_path,
                load_state_path=opts.earlystopping_load_state_path
            )

        self.wandb_logger = wandb.init(project=project, name=opts.task_name)
        if wandb_config is not None:
            self.wandb_logger.config.update({"model config": wandb_config})
        self.wandb_logger.config.update({
            "model_name": getattr(opts, "model_name", "None"),
            "model_type": getattr(opts, "model_type", "None"),
            "pretrained_model_path": getattr(opts, "pretrained_model_path", "None"),
            "epoch": opts.num_train_epochs,
            "learning_rate": opts.learning_rate,
            "batch_size": opts.per_gpu_train_batch_size,
        })

    def build_mixed_precision(self):
        # Mixed precision setup
        self.use_apex = False
        self.use_amp = False
        self.fp16_backend = None
        if self.opts.fp16:
            if self.opts.fp16_backend == "auto":
                self.fp16_backend = "amp" if _is_native_amp_available else "apex"
            else:
                self.fp16_backend = self.opts.fp16_backend
            self.logger.info(f"Using {self.fp16_backend} fp16 backend")
            if self.fp16_backend == "amp":
                self.use_amp = True
                self.scaler = torch.cuda.amp.GradScaler()
            else:
                if not is_apex_available():
                    msg = ("Using FP16 with APEX but APEX is not installed, "
                           "please refer to https://www.github.com/nvidia/apex.")
                    raise ImportError(msg)
                self.use_apex = True

    def build_adv_model(self):
        if self.opts.adv_type == 'fgm':
            adv_model = FGM(self.model,
                            emb_name=getattr(self.opts, 'adv_name', 'word_embeddings'),
                            epsilon=getattr(self.opts, 'adv_epsilon', 1.0))
        elif self.opts.adv_type == 'pgd':
            adv_model = PGD(self.model,
                            emb_name=getattr(self.opts, 'adv_name', 'word_embeddings'),
                            epsilon=getattr(self.opts, 'adv_epsilon', 1.0),
                            alpha=getattr(self.opts, 'adv_alpha', 0.3))
        elif self.opts.adv_type == 'awp':
            adv_model = AWP(self.model,
                            emb_name=getattr(self.opts, 'adv_name', 'word_embeddings'),
                            epsilon=getattr(self.opts, 'adv_epsilon', 1.0),
                            alpha=getattr(self.opts, 'adv_alpha', 0.3))
        elif self.opts.adv_type == 'freelb':
            adv_model = FreeLB(self.model,
                               adv_k=getattr(self.opts, 'adv_number', 3),
                               adv_lr=getattr(self.opts, 'adv_lr', 1e-2),
                               adv_init_mag=getattr(self.opts, 'adv_init_mag', 2e-2),
                               adv_max_norm=getattr(self.opts, 'adv_max_norm', 0.),
                               adv_norm_type=getattr(self.opts, 'adv_norm_type', 'l2'),
                               base_model=getattr(self.opts, 'base_model', 'bert'))
        elif self.opts.adv_type == 'vat':
            adv_model = ALUM(self.model,
                             adv_k=getattr(self.opts, 'adv_number', 1),
                             adv_lr=getattr(self.opts, 'adv_lr', 1e-2),
                             adv_var=getattr(self.opts, 'adv_var', 1e-5),
                             adv_alpha=getattr(self.opts, 'adv_alpha', 1.0),
                             adv_gamma=getattr(self.opts, 'adv_gamma', 1e-6),
                             adv_norm_type=getattr(self.opts, 'adv_norm_type', 'l2'),
                             base_model=getattr(self.opts, 'base_model', 'bert'))
        return adv_model

    def build_record_tracker(self, **kwargs):
        """
        build record object
        """
        self.records = {'result': {}, 'loss_meter': AverageMeter()}
        for key, value in kwargs.items():
            if key not in self.records:
                self.records[key] = value

    def reset_metrics(self):
        for metric in self.metrics:
            if hasattr(metric, 'reset'):
                metric.reset()

    def _param_optimizer(self, params, learning_rate, no_decay, weight_decay):
        # sourcery skip: inline-immediately-returned-variable, invert-any-all
        _params = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay,
             'lr': learning_rate},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             'lr': learning_rate},
        ]
        return _params

    def build_model_param_optimizer(self, model):
        """
        若需要对不同模型赋予不同学习率，则指定`base_model_name`,
        在`transformer`模块中，默认为`base_model_name=`base_model`.
        对于base_model使用learning_rate，
        其余统一使用other_learning_rate
        """
        no_decay = ["bias", 'LayerNorm.weight']
        optimizer_grouped_parameters = []
        if hasattr(model, self.opts.base_model_name) and self.opts.other_learning_rate != 0.0:
            msg = (f"The initial learning rate for model params : {self.opts.learning_rate} ,"
                   f"and {self.opts.other_learning_rate}"
                   )
            self.logger.info(msg)
            
            base_model = getattr(model, self.opts.base_model_name)
            base_model_param = list(base_model.named_parameters())
            base_model_param_ids = [id(p) for n, p in base_model_param]
            
            other_model_param = [(n, p) for n, p in model.named_parameters() if
                                 id(p) not in base_model_param_ids]
            
            optimizer_grouped_parameters.extend(
                self._param_optimizer(base_model_param, self.opts.learning_rate, no_decay, self.opts.weight_decay))
            optimizer_grouped_parameters.extend(
                self._param_optimizer(other_model_param, self.opts.other_learning_rate, no_decay,
                                      self.opts.weight_decay))
        else:
            all_model_param = list(model.named_parameters())
            optimizer_grouped_parameters.extend(
                self._param_optimizer(all_model_param, self.opts.learning_rate, no_decay, self.opts.weight_decay))
        
        return optimizer_grouped_parameters

    def build_optimizer(self, model):
        """
        Set up the optimizer.
        """
        optimizer_grouped_parameters = self.build_model_param_optimizer(model)
        optimizer_cls = Adafactor if self.opts.adafactor else AdamW
        
        if self.opts.adafactor:
            optimizer_cls = Adafactor
            optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
        
        else:
            optimizer = AdamW
            optimizer_kwargs = {
                "weight_decay": self.opts.weight_decay,
                "eps": self.opts.adam_epsilon,
            }

        optimizer = optimizer_cls(optimizer_grouped_parameters,
                                  lr=self.opts.learning_rate,
                                  **optimizer_kwargs)
        return optimizer

    def build_warmup_steps(self, num_training_steps):
        """
        Get number of steps used for a linear warmup.
        """
        if self.warmup_proportion < 0 or self.warmup_proportion > 1:
            raise ValueError("warmup_proportion must lie in range [0,1]")
        elif self.warmup_proportion > 0 and self.warmup_steps > 0:
            msg = ("Both warmup_ratio and warmup_steps given, "
                   "warmup_steps will override any effect of warmup_ratio during training")
            self.logger.info(msg)
        return self.warmup_steps if self.warmup_steps > 0 else math.ceil(num_training_steps * self.warmup_proportion)

    def build_lr_scheduler(self, num_training_steps):
        """
        the learning rate scheduler.
        """
        scheduler_function = get_lr_scheduler(self.opts.scheduler_type)
        warmup_steps = self.build_warmup_steps(num_training_steps)
        return scheduler_function(optimizer=self.optimizer, num_warmup_steps=warmup_steps,
                                  num_training_steps=num_training_steps)

    def freeze_to(self, n):
        """Freeze first n layers of model
        * **n** - Starting from initial layer, freeze all layers up to nth layer inclusively
        """
        layers = list(self.model.parameters())
        # Freeze up to n layers
        for param in layers[:n]:
            param.requires_grad = False
        for param in layers[n:]:
            param.requires_grad = True

    def build_train_dataloader(self, train_data):
        """
        Load train datasets
        """
        if isinstance(train_data, DataLoader):
            return train_data
        batch_size = self.opts.per_gpu_train_batch_size * max(1, self.device_num)
        sampler = train_data.sampler if hasattr(train_data, 'sampler') else RandomSampler(train_data)

        collate_fn = train_data.collate_fn if hasattr(train_data, 'collate_fn') else self.collate_fn
        return DataLoader(train_data, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn,
                          drop_last=self.opts.drop_last, num_workers=self.opts.num_workers)

    def build_eval_dataloader(self, dev_data):
        """
        Load eval datasets
        """
        if isinstance(dev_data, DataLoader):
            return dev_data
        batch_size = self.opts.per_gpu_eval_batch_size * max(1, self.device_num)
        sampler = dev_data.sampler if hasattr(dev_data, 'sampler') else SequentialSampler(dev_data)

        collate_fn = dev_data.collate_fn if hasattr(dev_data, 'collate_fn') else self.collate_fn
        return DataLoader(dev_data, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn,
                          num_workers=self.opts.num_workers)

    def build_test_dataloader(self, test_data):
        """
        Load test datasets
        """
        if isinstance(test_data, DataLoader):
            return test_data
        batch_size = self.opts.per_gpu_test_batch_size * max(1, self.device_num)
        sampler = test_data.sampler if hasattr(test_data, 'sampler') else SequentialSampler(test_data)

        collate_fn = test_data.collate_fn if hasattr(test_data, 'collate_fn') else None
        return DataLoader(test_data, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn,
                          num_workers=self.opts.num_workers)

    def build_batch_inputs(self, batch):
        """
        Sent all model inputs to the appropriate device (GPU on CPU)
        return:
         The inputs are in a dictionary format
        """
        return {
            key: (value.to(self.device) if ((key not in self.keys_to_ignore_on_gpu) and (value is not None)) else value)
            for key, value in batch.items()}

    def check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')
        if isinstance(loss, torch.Tensor) and torch.isnan(loss):
            import pdb
            pdb.set_trace()

    def build_writer(self):
        exp_dir = os.path.join(self.opts.output_dir, f'{self.prefix}_file_logs')
        self.writer = FileWriter(log_dir=exp_dir)

    def build_model_warp(self):
        # Mixed precision training with apex (torch < 1.6)
        if self.use_apex:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.opts.fp16_opt_level)
        # Multi-gpu training (should be after apex fp16 initialization)
        if self.device_num > 1:
            self.model = nn.DataParallel(self.model)

    def train_forward(self, batch):
        """
        Training forward
        """
        self.model.train()
        inputs = self.build_batch_inputs(batch)
        if self.use_amp:
            with autocast():
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)
        check_object_type(object=outputs, check_type=dict, name='outputs')
        if self.device_num > 1: outputs['loss'] = outputs['loss'].mean()
        return outputs

    def train_backward(self, loss):
        """
        Training backward
        """
        self.check_nan(loss)
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps
        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def train_update(self):
        # sourcery skip
        if self.use_amp:
            # AMP: gradients need unscaling
            self.scaler.unscale_(self.optimizer)
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                amp.master_params(self.optimizer) if self.use_apex else self.model.parameters(),
                self.max_grad_norm)
        optimizer_was_run = True
        if self.use_amp:
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scale_after = self.scaler.get_scale()
            optimizer_was_run = scale_before <= scale_after
        else:
            self.optimizer.step()
        if optimizer_was_run: self.scheduler.step()  # Update learning rate schedule
        self.model.zero_grad()  # Reset gradients to zero
        self.global_step += 1

    def train_adv(self, batch):
        if self.opts.adv_type == 'fgm':
            self.adv_model.attack()
            adv_outputs = self.train_forward(batch)
            adv_loss = adv_outputs['loss']
            self.train_backward(adv_loss)
        elif self.opts.adv_type in ['pgd', 'awp']:
            self.adv_model.backup_grad()
            for t in range(self.opts.adv_number):
                self.adv_model.attack(is_first_attack=(t == 0))
                if t != self.opts.adv_number - 1:
                    self.optimizer.zero_grad()
                else:
                    self.adv_model.restore_grad()
                adv_outputs = self.train_forward(batch)
                adv_loss = adv_outputs['loss']
                self.train_backward(adv_loss)
        self.adv_model.restore()

    def train_step(self, step, batch):
        if self.opts.adv_enable and self.opts.adv_type in ["freelb", "vat"]:
            loss = self.adv_model.attack(batch, gradient_accumulation_steps=self.gradient_accumulation_steps)
        else:
            outputs = self.train_forward(batch)
            loss = outputs['loss']
            self.train_backward(loss)
        should_save = False
        should_logging = False
        if self.opts.adv_enable:
            self.train_adv(batch)
        if (step + 1) % self.gradient_accumulation_steps == 0 or (
                self.gradient_accumulation_steps >= self.steps_in_epoch == (step + 1)
        ):
            self.train_update()
            should_logging = self.global_step % self.opts.logging_steps == 0
            should_save = self.global_step % self.opts.save_steps == 0
            self.records['loss_meter'].update(loss.item(), n=1)
            self.writer.add_scalar('loss/train_loss', loss.item(), self.global_step)
            if hasattr(self.scheduler, 'get_last_lr'):
                self.writer.add_scalar('learningRate/train_lr', self.scheduler.get_last_lr()[-1], self.global_step)
                self.wandb_logger.log({
                    "loss": loss.item(),
                    "lr": self.scheduler.get_last_lr()[-1]
                })
            return outputs, should_logging, should_save
        else:
            return None, should_logging, should_save

    # TODO 多机分布式训练
    def train(self, train_data, dev_data=None, resume_path=None, start_epoch=1, state_to_save=dict()):
        # sourcery skip: low-code-quality
        train_dataloader = self.build_train_dataloader(train_data)
        num_training_steps = len(train_dataloader) // self.gradient_accumulation_steps * self.num_train_epochs
        self.steps_in_epoch = len(train_dataloader)
        if self.scheduler is None:
            self.scheduler = self.build_lr_scheduler(num_training_steps)
        self.resume_from_checkpoint(resume_path=resume_path)
        self.build_model_warp()
        self.print_summary(len(train_data), num_training_steps)
        self.optimizer.zero_grad()
        seed_everything(self.opts.seed, verbose=False)  # Added here for reproducibility (even between python 2 and 3)
        if self.opts.logging_steps < 0:
            self.opts.logging_steps = len(train_dataloader) // self.gradient_accumulation_steps
            self.opts.logging_steps = max(1, self.opts.logging_steps)
        if self.opts.save_steps < 0:
            self.opts.save_steps = len(train_dataloader) // self.gradient_accumulation_steps
            self.opts.save_steps = max(1, self.opts.save_steps)
        self.build_record_tracker()
        self.reset_metrics()
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training', num_epochs=self.num_train_epochs)
        for epoch in range(start_epoch, int(self.num_train_epochs) + 1):
            pbar.epoch(current_epoch=epoch)
            for step, batch in enumerate(train_dataloader):
                outputs, should_logging, should_save = self.train_step(step, batch)
                if outputs is not None:
                    if self.opts.ema_enable:
                        self.model_ema.update(self.model)
                    pbar.step(step, {'loss': outputs['loss'].item()})
                if (self.opts.logging_steps > 0 and self.global_step > 0) and \
                        should_logging and self.opts.evaluate_during_training:
                    self.evaluate(dev_data)
                    if self.opts.ema_enable and self.model_ema is not None:
                        self.evaluate(dev_data, prefix_metric='ema')
                    if hasattr(self.writer, 'save'):
                        self.writer.save()
                if (self.opts.save_steps > 0 and self.global_step > 0) and should_save and self.model_checkpoint:
                    state = self.build_state_object(**state_to_save)
                    if self.opts.evaluate_during_training:
                        if self.model_checkpoint.monitor not in self.records['result']:
                            msg = ("There were expected keys in the eval result: "
                                   f"{', '.join(list(self.records['result'].keys()))}, "
                                   f"but get {self.model_checkpoint.monitor}."
                                   )
                            raise TypeError(msg)
                        self.model_checkpoint.step(
                            state=state,
                            current=self.records['result'][self.model_checkpoint.monitor]
                        )
                    else:
                        self.model_checkpoint.step(
                            state=state,
                            current=None
                        )
            # early_stopping
            if self.early_stopping:
                if self.early_stopping.monitor not in self.records['result']:
                    msg = ("There were expected keys in the eval result: "
                           f"{', '.join(list(self.records['result'].keys()))}, "
                           f"but get {self.early_stopping.monitor}."
                           )
                    raise TypeError(msg)
                self.early_stopping.step(
                    current=self.records['result'][self.early_stopping.monitor])
                if self.early_stopping.stop_training:
                    break
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if self.writer:
            self.writer.close()
        self.wandb_logger.finish()

    def build_state_object(self, **kwargs):
        """
        save state object
        """
        states = {
            'model': self.model.module if hasattr(self.model, "module") else self.model,
            'opts': self.opts,
            'optimizer': self.optimizer,
            'global_step': self.global_step,
            'vocab': self.tokenizer,
        }
        if self.scheduler is not None:
            states['scheduler'] = self.scheduler
        if self.use_amp:
            states['scaler'] = self.scaler
        for key, value in kwargs.items():
            if key not in states:
                states[key] = value
        return states

    def resume_from_checkpoint(self, resume_path=None):
        """
        Check if continuing training from a checkpoint
        """
        if resume_path is None:
            return
        optimizer_path = os.path.join(resume_path, OPTIMIZER_NAME)
        scheduler_path = os.path.join(resume_path, SCHEDULER_NAME)
        state_path = os.path.join(resume_path, TRAINER_STATE_NAME)
        model_path = os.path.join(resume_path, WEIGHTS_NAME)
        scaler_path = os.path.join(resume_path, SCALER_NAME)
        if is_file(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
        if is_file(scheduler_path):
            self.scheduler.load_state_dict(torch.load(scheduler_path))
        if is_file(state_path):
            state = torch.load(state_path)
            if self.model_checkpoint and hasattr(state, 'best_score'):
                self.model_checkpoint.best = state['best_score']
            del state
        if is_file(model_path):
            if self.use_amp and is_file(scaler_path):
                self.scaler.load_state_dict(torch.load(scaler_path))
            load_model(self.model, model_path, device=self.device)

    def print_summary(self, examples, t_total):
        # sourcery skip
        """
        print training parameters information
        """
        # self.logger.info("Training/evaluation parameters %s", self.opts)
        self.logger.info("***** Running training %s *****", self.opts.task_name)
        self.logger.info("  Options = %s", self.opts)
        self.logger.info("  Model type = %s", self.opts.model_type)
        self.logger.info("  Num examples = %d", examples)
        self.logger.info("  Num Epochs = %d", self.num_train_epochs)
        self.logger.info("  Instantaneous batch size per GPU = %d", self.opts.per_gpu_train_batch_size)
        self.logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                         self.opts.per_gpu_train_batch_size * self.device_num * self.gradient_accumulation_steps)
        self.logger.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        self.logger.info("  Total optimization steps = %d", t_total)
        self.logger.info("  Total Number of Parameters: %d" % sum(p.numel() for p in self.model.parameters()))
        # Calculating total number of trainable params
        self.logger.info("  Total Number of Trainable Parameters: %d " % sum(
            p.numel() for p in self.model.parameters() if p.requires_grad))

    def print_evaluate_result(self):
        """
        打印evaluation结果,
        """
        if len(self.records['result']) == 0:
            self.logger.warning("eval result record is empty")
        self.logger.info("***** Evaluating results of %s *****", self.opts.task_name)
        self.logger.info("  global step = %s", self.global_step)
        print_result = []
        for key, value in self.records['result'].items():
            if isinstance(value, (int, float)):
                print_result.insert(0, [key, value])
            else:
                print_result.append([key, value])
        for key, value in print_result:
            if isinstance(value, pd.DataFrame):
                self.logger.info(f" %s : \n %s", key, str(round(value, 5)))
            else:
                self.logger.info("  %s = %s", key, str(round(value, 5)))
                name = "_".join(key.split("_")[1:]) if "_" in key else key
                self.writer.add_scalar(f"{name}/{key}", value, int(self.global_step / self.opts.logging_steps))

        res = {f"{key}": value for key, value in print_result}
        self.wandb_logger.log(res)

    def save_predict_result(self, data, file_name, save_dir=None):
        """
        保存预测信息
        """
        if save_dir is None:
            save_dir = self.opts.output_dir
        elif not os.path.isdir(save_dir):
            save_dir = os.path.join(self.opts.output_dir, save_dir)
        file_path = os.path.join(save_dir, file_name)
        if ".pkl" in file_path:
            save_pickle(file_path=file_path, data=data)
        elif ".json" in file_path:
            json_to_text(file_path=file_path, data=data)
        else:
            raise ValueError("file type: expected one of (.pkl, .json)")

    def evaluate(self, dev_data, prefix_metric=None, save_dir=None, save_result=False, file_name=None):
        """
        Evaluate the model on a validation set
        """
        all_batch_list = []
        eval_dataloader = self.build_eval_dataloader(dev_data)
        self.build_record_tracker()
        self.reset_metrics()
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        for step, batch in enumerate(eval_dataloader):
            batch = self.predict_forward(batch)
            if 'loss' in batch and batch['loss'] is not None:
                self.records['loss_meter'].update(batch['loss'], n=1)
            all_batch_list.append(batch)
            pbar.step(step)
        self.records['result']['eval_loss'] = self.records['loss_meter'].avg
        self.update_metrics(all_batch_list, prefix_metric)
        self.print_evaluate_result()
        if save_result:
            if file_name is None:
                file_name = "dev_eval_results.pkl"
            self.save_predict_result(data=all_batch_list, file_name=file_name, save_dir=save_dir)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def update_metrics(self, all_batch_list, prefix):
        eval_data = self.build_batch_concat(all_batch_list, dim=0)
        prefix = '' if prefix is None else prefix + "_"
        for metric in self.metrics:
            metric.update(preds=eval_data['preds'], target=eval_data['target'])
            value = metric.value()
            if isinstance(value, float):
                self.records['result'][f'{prefix}eval_{metric.name()}'] = value
            elif isinstance(value, dict):
                self.records['result'].update({f"{prefix}eval_{k}": v for k, v in value.items()})
            elif value is None:
                self.logger.info(f"{metric.name()} value is None")
            else:
                msg = "metric value type: expected one of (float, dict,None)"
                raise ValueError(msg)

    def predict(self, test_data, save_result=True, file_name=None, save_dir=None):
        """
        test数据集预测
        """
        all_batch_list = []
        test_dataloader = self.build_test_dataloader(test_data)
        pbar = ProgressBar(n_total=len(test_dataloader), desc='Predicting')
        for step, batch in enumerate(test_dataloader):
            batch = self.predict_forward(batch)
            all_batch_list.append(batch)
            pbar.step(step)
        if save_result:
            if file_name is None:
                file_name = "test_predict_results.pkl"
            self.save_predict_result(data=all_batch_list, file_name=file_name, save_dir=save_dir)

    def predict_forward(self, batch):
        self.model.eval()
        inputs = self.build_batch_inputs(batch)
        with torch.no_grad():
            outputs = self.model(**inputs)
        if 'loss' in outputs and outputs['loss'] is not None:
            outputs['loss'] = outputs['loss'].mean().detach().item()
        outputs = {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in
                   outputs.items()}
        batch = {key: value for key, value in dict(batch, **outputs).items() if
                 key not in self.keys_to_ignore_on_result_save}
        return batch

    def build_batch_concat(self, all_batch_list):
        raise NotImplementedError('Method [build_batch_concat] should be implemented.')
