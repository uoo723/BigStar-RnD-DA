"""
Created on 2022/09/11
@author Sangwoo Han
"""
import os
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

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
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .. import base_trainer
from ..base_trainer import BaseTrainerModel
from ..datasets import LotteQADataset, collate_fn
from ..utils import AttrDict, filter_arguments, get_label_encoder
from .models import BaselineModel, BaselineModelWithMLAttention

BATCH = Tuple[Dict[str, torch.Tensor], torch.Tensor]


class BaselineTrainerModel(BaseTrainerModel):
    MODEL_HPARAMS: Iterable[str] = BaseTrainerModel.MODEL_HPARAMS + [
        "pretrained_model_name",
        "linear_size",
        "linear_dropout",
        "use_layernorm",
        "max_length",
        "model_name",
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
        model_name: str = "Baseline",
        linear_size: List[str] = [256],
        linear_dropout: float = 0.2,
        use_layernorm: bool = False,
        ls_alpha: Optional[float] = None,
        aug_filename: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.root_data_dir = root_data_dir
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.pretrained_model_name = pretrained_model_name
        self.model_name = model_name
        self.linear_size = linear_size
        self.linear_dropout = linear_dropout
        self.use_layernorm = use_layernorm
        self.ls_alpha = ls_alpha
        self.aug_filename = aug_filename
        self._le = None
        self._tokenizer = None
        self.save_hyperparameters(ignore=self.IGNORE_HPARAMS)

    @property
    def le(self) -> Union[LabelEncoder, MultiLabelBinarizer]:
        if self._le is None:
            dataset = LotteQADataset()
            if self.ls_alpha is not None:
                le_filename = "label_encoder_mlb.joblib"
                is_multilabel = True
            else:
                le_filename = "label_encoder.joblib"
                is_multilabel = False

            self._le = get_label_encoder(
                os.path.join(self.cache_dir, le_filename),
                dataset.y,
                is_multilabel=is_multilabel,
            )
        return self._le

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        return self._tokenizer

    def prepare_data(self) -> None:
        logger.info("Prepare data...")
        _ = self.le, self.tokenizer

    def setup_dataset(self, stage: Optional[str] = None) -> None:
        if stage == "predict":
            raise ValueError(f"{stage} stage is not supported")

        if stage == "fit" and self.train_dataset is None:
            train_dataset = LotteQADataset(aug_filename=self.aug_filename)
            val_dataset = LotteQADataset(mode="val")

            if self.valid_size == 1.0:
                self.val_ids = np.arange(len(val_dataset))
            else:
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

        model_cls = (
            BaselineModel
            if self.model_name == "Baseline"
            else BaselineModelWithMLAttention
        )

        self.model = model_cls(
            num_labels=len(self.le.classes_), **filter_arguments(hparams, model_cls)
        )

    def _get_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(self.le, LabelEncoder):
            loss = F.cross_entropy(inputs, targets)
        else:
            if self.ls_alpha is not None:
                targets = targets * (1 - self.ls_alpha) + self.ls_alpha / len(
                    self.le.classes_
                )
            loss = F.binary_cross_entropy_with_logits(inputs, targets)
        return loss

    def training_step(self, batch: BATCH, _) -> STEP_OUTPUT:
        batch_x, batch_y = batch

        outputs = self.model(batch_x)
        loss = self._get_loss(outputs, batch_y)
        self.log("loss/train", loss)
        return loss

    def _validation_and_test_step(
        self, batch: BATCH, is_val: bool = True
    ) -> Optional[STEP_OUTPUT]:
        batch_x, batch_y = batch
        outputs: torch.Tensor = self.model(batch_x)
        pred = outputs.argmax(dim=-1).cpu()

        if is_val:
            loss = self._get_loss(outputs, batch_y)
            self.log("loss/val", loss)

        return pred

    def _validation_and_test_epoch_end(
        self, outputs: EPOCH_OUTPUT, is_val: bool = True
    ) -> None:
        predictions = self.le.classes_[np.concatenate(outputs)]

        if is_val:
            gt = self.val_dataset.dataset.y[self.val_ids][: len(predictions)]
        else:
            gt = self.test_dataset.y[: len(predictions)]

        # f1_micro = prec_micro = recall_micro == accuracy
        f1_micro = f1_score(gt, predictions, average="micro", zero_division=0)

        prec_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            gt, predictions, average="macro", zero_division=0
        )

        (
            prec_weighted,
            recall_weighted,
            f1_weighted,
            _,
        ) = precision_recall_fscore_support(
            gt, predictions, average="weighted", zero_division=0
        )

        metrics = {
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "prec_macro": prec_macro,
            "recall_macro": recall_macro,
            "f1_weighted": f1_weighted,
            "prec_weighted": prec_weighted,
            "recall_weighted": recall_weighted,
        }

        if is_val:
            self.log_dict(
                {f"val/{self.early_criterion}": metrics.pop(self.early_criterion)},
                prog_bar=True,
            )
            self.log_dict({f"val/{k}": v for k, v in metrics.items()})
        else:
            self.log_dict({f"test/{k}": v for k, v in metrics.items()})

    def validation_step(self, batch: BATCH, _) -> Optional[STEP_OUTPUT]:
        return self._validation_and_test_step(batch, is_val=True)

    def test_step(self, batch: BATCH, _) -> Optional[STEP_OUTPUT]:
        return self._validation_and_test_step(batch, is_val=False)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._validation_and_test_epoch_end(outputs, is_val=True)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._validation_and_test_epoch_end(outputs, is_val=False)


def check_args(args: AttrDict) -> None:
    valid_early_criterion = [
        "f1_micro",
        "prec_macro",
        "recall_macro",
        "f1_weighted",
        "prec_weighted",
        "recall_weighted",
        "loss",
    ]
    valid_model_name = ["Baseline", "BaselineWithMLAttention"]
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
        metrics=[
            "f1_micro",
            "f1_macro",
            "prec_macro",
            "recall_macro",
            "f1_weighted",
            "prec_weighted",
            "recall_weighted",
        ],
        trainer=trainer,
        is_hptuning=is_hptuning,
    )


def predict(args: AttrDict) -> Any:
    pass
