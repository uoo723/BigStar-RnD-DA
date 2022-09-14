"""
Created on 2022/06/07
@author Sangwoo Han
"""
import os
import re
from abc import ABC, abstractmethod
from ast import literal_eval
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import optuna
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
import torch.cuda
import torch.nn as nn
import torch.optim
from deepspeed.ops.adam import DeepSpeedCPUAdam
from logzero import logger
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
from optuna import Trial
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from ruamel.yaml import YAML
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from transformers import get_scheduler

from .callbacks import MLFlowExceptionCallback, StochasticWeightAveraging
from .optimizers import DenseSparseAdamW
from .utils import AttrDict, filter_arguments, set_seed


def get_run(log_dir: str, run_id: str) -> Run:
    client = MlflowClient(os.environ.get("MLFLOW_TRACKING_URI", log_dir))
    run = client.get_run(run_id)
    return run


def _get_single_ckpt_path(ckpt_path: str) -> str:
    if os.path.isdir(ckpt_path):
        basename, ext = os.path.splitext(os.path.basename(ckpt_path))
        new_ckpt_path = os.path.join(os.path.dirname(ckpt_path), f"{basename}.ds{ext}")
        convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, new_ckpt_path)
        return new_ckpt_path
    return ckpt_path


def get_ckpt_path(
    log_dir: str, run_id: str, load_best: bool = False, use_deepspeed: bool = False
) -> Optional[str]:
    run = get_run(log_dir, run_id)
    ckpt_root_dir = os.path.join(log_dir, run.info.experiment_id, run_id, "checkpoints")
    ckpt_path = os.path.join(ckpt_root_dir, "last.ckpt")

    if not os.path.exists(ckpt_path):
        return None

    if not load_best:
        return _get_single_ckpt_path(ckpt_path) if use_deepspeed else ckpt_path

    ckpt_path = _get_single_ckpt_path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    key = [k for k in ckpt["callbacks"].keys() if k.startswith("ModelCheckpoint")][0]
    ckpt_path = ckpt["callbacks"][key]["best_model_path"]
    ckpt_path = _get_single_ckpt_path(ckpt_path) if use_deepspeed else ckpt_path

    return ckpt_path


def _get_gpu_info(num_gpus: int) -> List[str]:
    return [f"{i}: {torch.cuda.get_device_name(i)}" for i in range(num_gpus)]


def _get_run_data(
    log_dir: str, run_id: str, data_type: str = "params"
) -> Dict[str, Any]:
    assert data_type in ["params", "tags"]

    run = get_run(log_dir, run_id)
    ret_params: Dict[str, Any] = {}
    params: Dict[str, Any] = run.data.params if data_type == "params" else run.data.tags

    for k, v in params.items():
        try:
            ret_params[k] = literal_eval(v)
        except Exception:
            ret_params[k] = v  # str type
    return ret_params


def get_model_hparams(
    log_dir: str, run_id: str, model_hparams: Iterable[str]
) -> Dict[str, Any]:
    params: Dict[str, Any] = _get_run_data(log_dir, run_id)
    return {k: v for k, v in params.items() if k in model_hparams}


def get_run_tags(log_dir: str, run_id: str) -> Dict[str, Any]:
    return _get_run_data(log_dir, run_id, data_type="tags")


def load_model_state(
    model: nn.Module,
    ckpt_path: str,
    substitution: Optional[Tuple] = None,
    load_average_model: bool = True,
) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    substitution = substitution or ("", "")

    swa_callback_key = None
    callbacks: Dict[str, Any] = ckpt["callbacks"]
    for key in callbacks.keys():
        if "StochasticWeightAveraging" in key:
            swa_callback_key = key
            break

    state_dict: Dict[str, torch.Tensor] = ckpt["state_dict"]

    if (
        load_average_model
        and swa_callback_key is not None
        and "average_model" in callbacks[swa_callback_key]
    ):
        avg_state_dict: Dict[str, torch.Tensor] = callbacks[swa_callback_key][
            "average_model"
        ]
        avg_state_dict.pop("models_num")
        state_dict.update(avg_state_dict)

    state_dict = OrderedDict(
        zip(
            [re.sub(*substitution, key) for key in state_dict.keys()],
            state_dict.values(),
        )
    )

    model.load_state_dict(state_dict)


def _get_optimizer(
    model: nn.Module,
    optim_name: str = "adamw",
    lr: float = 1e-3,
    decay: float = 0,
    use_deepspeed: bool = False,
) -> Optimizer:
    if use_deepspeed:
        assert optim_name == "adamw", "If set use_deepspeed, adamw is only allowed"

    no_decay = ["bias", "LayerNorm.weight"]

    param_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": decay,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
    ]

    if optim_name == "adamw":
        optim = (
            DeepSpeedCPUAdam(param_groups)
            if use_deepspeed
            else DenseSparseAdamW(param_groups)
        )
    elif optim_name == "sgd":
        optim = torch.optim.SGD(param_groups)
    else:
        raise ValueError(f"Optimizer {optim_name} is not supported")

    return optim


def _get_scheduler(
    optimizer: Optimizer,
    num_epochs: int,
    train_size: int,
    batch_size: int,
    accumulation_step: int = 1,
    scheduler_type: Optional[str] = None,
    scheduler_warmup: Optional[Union[float, int]] = None,
) -> Optional[_LRScheduler]:
    if scheduler_type is None:
        return

    step_size = batch_size * accumulation_step
    num_training_steps = (train_size + step_size - 1) // step_size * num_epochs

    if scheduler_warmup is not None:
        if isinstance(scheduler_warmup, float):
            if not 0 <= scheduler_warmup <= 1:
                raise ValueError(f"scheduler_warmup must be 0 <= scheduler_warmup <= 1")
            num_warmup_steps = int(scheduler_warmup * num_training_steps)
        elif isinstance(scheduler_warmup, int):
            if scheduler_warmup > num_training_steps:
                raise ValueError(
                    f"scheduler_warmup must be less than num_training_steps"
                )
            num_warmup_steps = scheduler_warmup
        else:
            raise ValueError(f"num_warmup_steps must be float|int")
    else:
        num_warmup_steps = None

    return get_scheduler(
        scheduler_type,
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )


class BaseTrainerModel(pl.LightningModule, ABC):
    IGNORE_HPARAMS: List[str] = [
        "model_name",
        "dataset_name",
        "num_gpus",
        "num_workers",
        "experiment_name",
        "run_name",
        "run_script",
        "trial",
        "enable_trial_pruning",
        "is_hptuning",
        "tags",
        "log_dir",
        "data_dir",
        "run_id",
        "reset_early",
        "ckpt_path",
        "use_deepspeed",
    ]

    MODEL_HPARAMS: Iterable[str] = []

    def __init__(
        self,
        num_gpus: int = 1,
        model_name: str = "model_name",
        dataset_name: str = "dataset_name",
        seed: Optional[int] = None,
        num_workers: int = 4,
        train_batch_size: int = 128,
        test_batch_size: int = 256,
        valid_size: float = 0.2,
        early: int = 10,
        reset_early: bool = False,
        ckpt_path: Optional[str] = None,
        use_deepspeed: bool = False,
        early_criterion: str = "f1",
        eval_step: int = 100,
        optim_name: str = "adamw",
        lr: float = 1e-3,
        decay: float = 1e-2,
        num_epochs: int = 40,
        accumulation_step: int = 1,
        gradient_max_norm: Optional[float] = None,
        swa_warmup: int = 0,
        scheduler_type: Optional[str] = None,
        scheduler_warmup: Optional[float] = None,
        is_hptuning: bool = False,
        trial: Optional[Trial] = None,
        enable_trial_pruning: bool = False,
        experiment_name: Optional[str] = None,
        run_id: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Iterable[Tuple[str, Any]] = [],
        run_script: Optional[str] = None,
        log_dir: str = "./logs",
        data_dir: str = "./data",
    ) -> None:
        super().__init__()
        self.num_gpus = num_gpus
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.seed = seed
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.valid_size = valid_size
        self.early = early
        self.reset_early = reset_early
        self.ckpt_path = ckpt_path
        self.use_deepspeed = use_deepspeed
        self.early_criterion = early_criterion
        self.eval_step = eval_step
        self.optim_name = optim_name
        self.lr = lr
        self.decay = decay
        self.num_epochs = num_epochs
        self.accumulation_step = accumulation_step
        self.gradient_max_norm = gradient_max_norm
        self.swa_warmup = swa_warmup
        self.scheduler_type = scheduler_type
        self.scheduler_warmup = scheduler_warmup
        self.is_hptuning = is_hptuning
        self.trial = trial
        self.enable_trial_pruning = enable_trial_pruning
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.run_name = run_name
        self.tags = tags
        self.run_script = run_script
        self.log_dir = log_dir
        self.data_dir = data_dir

        self._logged = False

        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    @abstractmethod
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    @abstractmethod
    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    @abstractmethod
    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    @abstractmethod
    def prepare_data(self) -> None:
        pass

    @abstractmethod
    def setup_dataset(self, stage: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def setup_model(self, stage: Optional[str] = None) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.setup_dataset(stage)
        self.setup_model(stage)

        if not self._logged:
            logger.info(f"Model: {self.model_name}")
            logger.info(f"Dataset: {self.dataset_name}")

            if self.train_dataset:
                logger.info(f"# train dataset: {len(self.train_dataset):,}")

            if self.val_dataset:
                logger.info(f"# valid dataset: {len(self.val_dataset):,}")

            if self.test_dataset:
                logger.info(f"# test dataset: {len(self.test_dataset):,}")

            self._logged = True

        if self.ckpt_path:
            logger.info(f"Load model weights from ({self.ckpt_path})")
            load_model_state(self.model, self.ckpt_path, substitution=(r"^model\.", ""))

    def configure_optimizers(self):
        optimizer = _get_optimizer(
            self.model, self.optim_name, self.lr, self.decay, self.use_deepspeed
        )

        scheduler = _get_scheduler(
            optimizer,
            self.num_epochs,
            len(self.train_dataloader().dataset),
            self.train_dataloader().batch_size,
            self.accumulation_step,
            self.scheduler_type,
            self.scheduler_warmup,
        )

        if scheduler is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_fit_start(self) -> None:
        experiment: MlflowClient = self.logger.experiment

        experiment.set_tag(self.logger.run_id, "run_id", self.logger.run_id)
        experiment.set_tag(self.logger.run_id, "host", os.uname()[1])
        experiment.set_tag(self.logger.run_id, "model_name", self.model_name)
        experiment.set_tag(self.logger.run_id, "dataset_name", self.dataset_name)
        experiment.set_tag(self.logger.run_id, "use_deepspeed", self.use_deepspeed)

        if self.run_id is not None:
            experiment.set_tag(self.logger.run_id, "resume", self.run_id)

        for k, v in self.tags:
            experiment.set_tag(self.logger.run_id, k, v)

        logger.info(f"experiment: {self.experiment_name}")
        logger.info(f"run_name: {self.run_name}")
        logger.info(f"run_id: {self.logger.run_id}")

        if self.run_script is not None:
            experiment.log_artifact(self.logger.run_id, self.run_script, "scripts")

        if self.num_gpus >= 1:
            gpu_info = _get_gpu_info(self.num_gpus)
            experiment.set_tag(self.logger.run_id, "GPU info", ", ".join(gpu_info))

        if self.run_id is not None and self.reset_early:
            for callback in self.trainer.callbacks:
                if isinstance(callback, EarlyStopping):
                    callback.wait_count = 0
                    break

    def should_prune(self, value: float) -> None:
        experiment: MlflowClient = self.logger.experiment
        if self.trial is not None and self.enable_trial_pruning:
            self.trial.report(value, self.global_step)
            if self.trial.should_prune():
                experiment.set_tag(self.logger.run_id, "pruned", True)
                raise optuna.TrialPruned


def check_args(
    args: AttrDict,
    valid_early_criterion: List[str],
    valid_model_name: List[str],
    valid_dataset_name: List[str],
) -> None:
    # if args.mode in ["test", "predict"]:
    #     assert args.run_id is not None, f"run_id must be specified in mode {args.mode}"

    assert (
        args.early_criterion in valid_early_criterion
    ), f"early_criterion must be one of {valid_early_criterion}"

    assert (
        args.model_name in valid_model_name
    ), f"model_name must be one of {valid_model_name}"

    assert (
        args.dataset_name in valid_dataset_name
    ), f"dataset_name must be one of {valid_dataset_name}"

    assert type(args.valid_size) in [float, int], "valid size must be int or float"

    if type(args.valid_size) == int:
        assert args.valid_size > 0, "valid size must be greater than 0"

    if type(args.valid_size) == float:
        assert 0 < args.valid_size < 1, "valid size must be 0 < valid_size < 1"


def init_run(args: AttrDict) -> None:
    if args.seed is not None:
        logger.info(f"seed: {args.seed}")
        set_seed(args.seed)

    args.device = torch.device("cpu" if args.no_cuda else "cuda")
    args.num_gpus = torch.cuda.device_count()

    yaml = YAML(typ="safe")
    if args.model_cnf is not None:
        model_cnf = yaml.load(Path(args.model_cnf))
        args.update(model_cnf["model"])

    if args.data_cnf is not None:
        data_cnf = yaml.load(Path(args.data_cnf))
        args.update(data_cnf["dataset"])


def train(
    args: AttrDict,
    TrainerModel: Type[BaseTrainerModel],
    is_hptuning: bool = False,
    trial: Optional[Trial] = None,
    enable_trial_pruning: bool = False,
) -> Tuple[float, pl.Trainer]:
    mlf_logger = pl_loggers.MLFlowLogger(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        save_dir=args.log_dir,
    )

    monitor = (
        "loss/val" if args.early_criterion == "loss" else f"val/{args.early_criterion}"
    )

    mode = "min" if args.early_criterion in ["loss"] else "max"

    callbacks = []
    callbacks.append(EarlyStopping(monitor=monitor, patience=args.early, mode=mode))
    callbacks.append(MLFlowExceptionCallback())
    callbacks.append(
        ModelCheckpoint(
            monitor=monitor,
            filename=f"epoch={{epoch:02d}}-{monitor.split('/')[-1]}={{{monitor}:.4f}}",
            mode=mode,
            save_top_k=1,
            auto_insert_metric_name=False,
            save_last=True,
        )
    )
    if args.swa_warmup > 0:
        callbacks.append(StochasticWeightAveraging(args.swa_warmup))

    ckpt_path = (
        get_ckpt_path(args.log_dir, args.run_id, load_best=args.load_best)
        if args.run_id
        else None
    )

    if args.load_only_weights:
        args.ckpt_path = ckpt_path
        ckpt_path = None

    if args.run_id is not None:
        hparams = get_model_hparams(
            args.log_dir, args.run_id, TrainerModel.MODEL_HPARAMS
        )
        args.update(hparams)

    trainer_model = TrainerModel(
        is_hptuning=is_hptuning,
        trial=trial,
        enable_trial_pruning=enable_trial_pruning,
        **filter_arguments(args, TrainerModel),
    )

    trainer = pl.Trainer(
        default_root_dir=args.log_dir,
        gpus=args.num_gpus,
        precision=16 if args.mp_enabled else 32,
        max_epochs=args.num_epochs,
        gradient_clip_val=args.gradient_max_norm,
        accumulate_grad_batches=args.accumulation_step,
        val_check_interval=args.eval_step,
        callbacks=callbacks,
        logger=mlf_logger,
        strategy="deepspeed_stage_2_offload" if args.use_deepspeed else None,
    )

    try:
        trainer.fit(trainer_model, ckpt_path=ckpt_path)
    except optuna.TrialPruned:
        pass

    args.run_id = mlf_logger.run_id
    args.load_model_only_weights = False
    args.ckpt_path = None

    if args.save_run_id_path is not None:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.save_run_id_path)), exist_ok=True
        )
        with open(args.save_run_id_path, "w", encoding="utf8") as f:
            f.write(args.run_id)

    model_checkpoint: ModelCheckpoint = callbacks[2]
    best_score: Optional[torch.Tensor] = model_checkpoint.best_model_score
    best_score = best_score.item() if best_score else 0

    return best_score, trainer


def test(
    args: AttrDict,
    TrainerModel: Type[BaseTrainerModel],
    metrics: List[str],
    trainer: Optional[pl.Trainer] = None,
    is_hptuning: bool = False,
) -> Dict[str, float]:
    # assert args.run_id is not None, "run_id must be specified"

    ckpt_path = None

    if args.run_id is not None:
        ckpt_path = get_ckpt_path(
            args.log_dir, args.run_id, load_best=not args.load_last
        )
        hparams = get_model_hparams(
            args.log_dir, args.run_id, TrainerModel.MODEL_HPARAMS
        )
        args.update(hparams)

    if trainer is None:
        trainer_model = TrainerModel(
            is_hptuning=is_hptuning, **filter_arguments(args, TrainerModel)
        )

        trainer_model.setup(stage="test")

        if ckpt_path is not None:
            load_model_state(trainer_model, ckpt_path)

        trainer = pl.Trainer(
            gpus=args.num_gpus,
            precision=16 if args.mp_enabled else 32,
            enable_model_summary=False,
            max_epochs=1,
            logger=False,
        )
    else:
        if ckpt_path is not None:
            train_model = trainer.model
            load_model_state(train_model, ckpt_path)

    results = trainer.test(trainer_model, verbose=False)

    if results is not None:
        results = results[0]
        msg = "\n" + "\n".join([f"{m}: {results['test/' + m]:.4f}" for m in metrics])
        logger.info(msg)

    return results or {}


def predict(
    args: AttrDict,
    TrainerModel: Type[BaseTrainerModel],
    trainer: Optional[pl.Trainer] = None,
) -> Any:
    assert args.mode == "predict", "mode must be predict"
    assert args.run_id is not None, "run_id must be specified"

    ckpt_path = get_ckpt_path(args.log_dir, args.run_id, load_best=True)

    if trainer is None:
        trainer_model = TrainerModel(**filter_arguments(args, TrainerModel))

        swa_warmup = int(get_run(args.log_dir, args.run_id).data.params["swa_warmup"])
        callbacks = []
        if swa_warmup > 0:
            callbacks.append(StochasticWeightAveraging(swa_warmup))

        trainer = pl.Trainer(
            gpus=args.num_gpus,
            precision=16 if args.mp_enabled else 32,
            enable_model_summary=False,
            logger=False,
            callbacks=callbacks,
        )
    else:
        trainer_model = trainer.lightning_module

    predictions = trainer.predict(trainer_model, ckpt_path=ckpt_path)

    return predictions
