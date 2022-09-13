"""
Created on 2022/09/11
@author Sangwoo Han
"""
import os
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from logzero import logger
from optuna import Trial
from pytorch_lightning.utilities.types import (
    EPOCH_OUTPUT,
    EVAL_DATALOADERS,
    STEP_OUTPUT,
    TRAIN_DATALOADERS,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from .. import base_trainer
from ..base_trainer import BaseTrainerModel
from ..datasets import LotteQADataset, collate_fn
from ..utils import AttrDict, filter_arguments, get_label_encoder
from .models import BaselineModel

BATCH = Tuple[Dict[str, torch.Tensor], torch.Tensor]


class BaselineTrainerModel(BaseTrainerModel):
    MODEL_HPARAMS: Iterable[str] = BaseTrainerModel.MODEL_HPARAMS + [
        "pretrained_model_name",
        "linear_size",
        "linear_dropout",
        "use_layernorm",
        "max_length",
    ]

    IGNORE_HPARAMS: List[str] = BaseTrainerModel.IGNORE_HPARAMS + [
        "root_data_dir",
        "cache_dir",
    ]

    def __init__(
        self,
        root_data_dir: str = "./data",
        cache_dir: str = "./cache_dir",
        max_length: int = 30,
        pretrained_model_name: str = "monologg/koelectra-base-v3-discriminator",
        linear_size: List[str] = [256],
        linear_dropout: float = 0.2,
        use_layernorm: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.root_data_dir = root_data_dir
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.pretrained_model_name = pretrained_model_name
        self.linear_size = linear_size
        self.linear_dropout = linear_dropout
        self.use_layernorm = use_layernorm
        self.save_hyperparameters(ignore=self.IGNORE_HPARAMS)

    def prepare_data(self) -> None:
        logger.info("Prepare data...")
        dataset = LotteQADataset()
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        self.le = get_label_encoder(
            os.path.join(self.cache_dir, "label_encoder"), dataset.y
        )

    def setup_dataset(self, stage: Optional[str] = None) -> None:
        if stage == "predict":
            raise ValueError(f"{stage} stage is not supported")

        if stage == "fit" and self.train_dataset is None:
            train_dataset = LotteQADataset()
            val_dataset = LotteQADataset(mode="val")

            _, self.val_ids = train_test_split(
                np.arange(len(val_dataset)),
                test_size=self.valid_size,
                random_state=self.seed,
            )

            self.train_dataset = train_dataset
            self.val_dataset = Subset(val_dataset, self.val_ids)

        if stage == "test" and self.test_dataset is None:
            self.test_dataset = LotteQADataset(mode="test")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device != torch.device("cpu"),
            collate_fn=partial(
                collate_fn,
                tokenizer=self.tokenizer,
                le=self.le,
                max_length=self.max_length,
            ),
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.device != torch.device("cpu"),
            collate_fn=partial(
                collate_fn,
                tokenizer=self.tokenizer,
                le=self.le,
                max_length=self.max_length,
            ),
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.device != torch.device("cpu"),
            collate_fn=partial(
                collate_fn,
                tokenizer=self.tokenizer,
                le=self.le,
                max_length=self.max_length,
            ),
        )

    def setup_model(self, stage: Optional[str] = None) -> None:
        if self.model is not None:
            return

        hparams = {param: getattr(self, param) for param in self.MODEL_HPARAMS}

        self.model = BaselineModel(
            num_labels=len(self.le.classes_), **filter_arguments(hparams, BaselineModel)
        )

    def training_step(self, batch: BATCH, _) -> STEP_OUTPUT:
        batch_x, batch_y = batch

        outputs = self.model(batch_x)
        loss = F.cross_entropy(outputs, batch_y)
        self.log("loss/train", loss)
        return loss

    def _validation_and_test_step(
        self, batch: BATCH, is_val: bool = True
    ) -> Optional[STEP_OUTPUT]:
        batch_x, batch_y = batch
        outputs: torch.Tensor = self.model(batch_x)
        pred = outputs.argmax(dim=-1).cpu()

        if is_val:
            loss = F.cross_entropy(outputs, batch_y)
            self.log("loss/val", loss)

        return pred

    def _validation_and_test_epoch_end(
        self, outputs: EPOCH_OUTPUT, is_val: bool = True
    ) -> None:
        predictions = np.concatenate(outputs)

        if is_val:
            gt = self.le.transform(
                self.val_dataset.dataset.y[self.val_ids][: len(predictions)]
            )
        else:
            gt = self.le.transform(self.test_dataset.y)[: len(predictions)]

        acc = accuracy_score(gt, predictions)
        f1 = f1_score(gt, predictions, average="micro")

        if is_val:
            self.log_dict({"val/f1": f1, "val/acc": acc}, prog_bar=True)
        else:
            self.log_dict({"test/f1": f1, "test/acc": acc})

    def validation_step(self, batch: BATCH, _) -> Optional[STEP_OUTPUT]:
        return self._validation_and_test_step(batch, is_val=True)

    def test_step(self, batch: BATCH, _) -> Optional[STEP_OUTPUT]:
        return self._validation_and_test_step(batch, is_val=False)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._validation_and_test_epoch_end(outputs, is_val=True)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._validation_and_test_epoch_end(outputs, is_val=False)


def check_args(args: AttrDict) -> None:
    valid_early_criterion = ["f1", "acc"]
    valid_model_name = ["Baseline"]
    valid_dataset_name = ["LotteQA"]
    base_trainer.check_args(
        args, valid_early_criterion, valid_model_name, valid_dataset_name
    )


def init_run(args: AttrDict) -> None:
    base_trainer.init_run(args)


def train(
    args: AttrDict,
    is_hptuning: bool = False,
    trial: Optional[Trial] = None,
    enable_trial_pruning: bool = False,
) -> Tuple[float, pl.Trainer]:
    return base_trainer.train(
        args,
        BaselineTrainerModel,
        is_hptuning=is_hptuning,
        trial=trial,
        enable_trial_pruning=enable_trial_pruning,
    )


def test(
    args: AttrDict, trainer: Optional[pl.Trainer] = None, is_hptuning: bool = False
) -> Dict[str, float]:
    return base_trainer.test(
        args,
        BaselineTrainerModel,
        metrics=["f1", "acc"],
        trainer=trainer,
        is_hptuning=is_hptuning,
    )


def predict(args: AttrDict) -> Any:
    pass
