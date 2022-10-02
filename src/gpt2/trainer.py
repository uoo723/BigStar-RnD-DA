"""
Created on 2022/10/01
@author Sangwoo Han
"""
import os
from itertools import chain
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from logzero import logger
from optuna import Trial
from pytorch_lightning.utilities.types import (
    EVAL_DATALOADERS,
    STEP_OUTPUT,
    TRAIN_DATALOADERS,
)
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .. import base_trainer
from ..base_trainer import BaseTrainerModel
from ..datasets import LotteQADataset, collate_fn
from ..utils import AttrDict, get_label_encoder, get_num_batches

BATCH = Tuple[Dict[str, torch.Tensor], torch.Tensor]


class GPT2TrainerModel(BaseTrainerModel):
    MODEL_HPARAMS: Iterable[str] = BaseTrainerModel.MODEL_HPARAMS + [
        "pretrained_model_name",
    ]

    IGNORE_HPARAMS: List[str] = BaseTrainerModel.IGNORE_HPARAMS + [
        "root_data_dir",
        "cache_dir",
    ]

    def __init__(
        self,
        root_data_dir: str = "./data",
        cache_dir: str = "./cache_dir",
        pretrained_model_name: str = "skt/kogpt2-base-v2",
        block_size: int = 1024,
        aug_filename: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.root_data_dir = root_data_dir
        self.cache_dir = cache_dir
        self.block_size = block_size
        self.pretrained_model_name = pretrained_model_name
        self.aug_filename = aug_filename
        self._le = None
        self._tokenizer = None
        self.save_hyperparameters(ignore=self.IGNORE_HPARAMS)

    @property
    def le(self) -> Union[LabelEncoder, MultiLabelBinarizer]:
        if self._le is None:
            dataset = LotteQADataset()
            self._le = get_label_encoder(
                os.path.join(self.cache_dir, "label_encoder.joblib"),
                dataset.y,
            )
        return self._le

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            tokenizer_kwargs = (
                dict(
                    bos_token="</s>",
                    eos_token="</s>",
                    unk_token="<unk>",
                    pad_token="<pad>",
                    mask_token="<mask>",
                )
                if self.pretrained_model_name == "skt/kogpt2-base-v2"
                else {}
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_model_name,
                **tokenizer_kwargs,
            )
        return self._tokenizer

    def prepare_data(self) -> None:
        logger.info("Prepare data...")
        _ = self.le, self.tokenizer

    def setup_dataset(self, stage: Optional[str] = None) -> None:
        if stage == "predict":
            raise ValueError(f"{stage} stage is not supported")

        if stage == "fit" and self.train_dataset is None:
            assert self.pretrained_model_name == "skt/kogpt2-base-v2"
            logger.info("Setup Dataset...")
            df = LotteQADataset(aug_filename=self.aug_filename).df

            # <unused1> ~ <unused13>
            label_token_id_map = {
                self.le.classes_[i]: 10 + i for i in range(self.le.classes_.size)
            }

            # <unused0>
            sep_token_id = 9

            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            inputs = self.tokenizer(df["발화문"].tolist())
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            labels = df["인텐트"].tolist()

            # flatten input_ids (i.e. single long sequence of all texts)
            input_ids = list(
                chain(
                    *[
                        [label_token_id_map[labels[i]]]
                        + [sep_token_id]
                        + ids
                        + [self.tokenizer.eos_token_id]
                        for i, ids in enumerate(inputs["input_ids"])
                    ]
                )
            )

            # Generate chunks of block size
            total_length = len(input_ids)
            num_blocks = get_num_batches(self.block_size, total_length)
            input_ids = [
                input_ids[i * self.block_size : (i + 1) * self.block_size]
                for i in range(num_blocks)
            ]
            num_pads = self.block_size - len(input_ids[-1])
            input_ids[-1].extend([self.tokenizer.pad_token_id] * num_pads)
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.ones(
                len(input_ids), self.block_size, dtype=torch.long
            )
            if num_pads > 0:
                attention_mask[-1, -num_pads:] = 0

            self.train_dataset = TensorDataset(input_ids, attention_mask)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.device != torch.device("cpu"),
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplemented

    def test_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplemented

    def setup_model(self, stage: Optional[str] = None) -> None:
        if self.model is not None:
            return

        self.model = AutoModelForCausalLM.from_pretrained(self.pretrained_model_name)

    def training_step(self, batch: BATCH, _) -> STEP_OUTPUT:
        input_ids, attention_mask = batch

        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
        )

        self.log("loss/train", outputs.loss)

        return outputs.loss


def check_args(args: AttrDict) -> None:
    valid_early_criterion = ["loss"]
    valid_model_name = ["GPT2"]
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
        GPT2TrainerModel,
        is_hptuning=is_hptuning,
        trial=trial,
        enable_trial_pruning=enable_trial_pruning,
    )


def test(
    args: AttrDict, trainer: Optional[pl.Trainer] = None, is_hptuning: bool = False
) -> Dict[str, float]:
    raise NotImplemented


def predict(args: AttrDict) -> Any:
    raise NotImplemented
