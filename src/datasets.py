"""
Created on 2022/09/10
@author Sangwoo Han
"""
import os
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class LotteQADataset(Dataset):
    def __init__(
        self,
        root_data_dir: str = "./data",
        mode: str = "train",
        df: Optional[pd.DataFrame] = None,
        aug_filename: Optional[str] = None,
    ) -> None:
        assert mode in {"train", "val", "test"}

        self._x = None
        self._y = None

        if aug_filename is not None:
            df = pd.read_csv(
                os.path.join(root_data_dir, aug_filename), low_memory=False
            )
            assert "발화문" in df.columns and "인텐트" in df.columns
            self.df = df[["발화문", "인텐트"]].drop_duplicates().reset_index(drop=True)
            return

        elif df is not None:
            self.df = df
            return

        if mode == "train":
            df = pd.read_csv(os.path.join(root_data_dir, "train.csv"), low_memory=False)
        elif mode == "val":
            df = pd.read_csv(
                os.path.join(root_data_dir, "validation.csv"), low_memory=False
            )
        elif mode == "test":
            df = pd.read_csv(os.path.join(root_data_dir, "test.csv"), low_memory=False)

        self.df = (
            df[df["QA여부"] == "q"][["발화문", "인텐트"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def __getitem__(self, index) -> Tuple[str, str]:
        return tuple(self.df.iloc[index][["발화문", "인텐트"]].tolist())

    def __len__(self) -> int:
        return len(self.df)

    @property
    def x(self) -> np.ndarray:
        if self._x is None:
            self._x = self.df["발화문"].to_numpy()
        return self._x

    @property
    def y(self) -> np.ndarray:
        if self._y is None:
            self._y = self.df["인텐트"].to_numpy()
        return self._y


def collate_fn(
    batch: Iterable[Tuple[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    le: Union[LabelEncoder, MultiLabelBinarizer],
    max_length: int = 30,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    x = [b[0] for b in batch]
    y = [b[1] for b in batch]
    inputs = tokenizer(
        x,
        max_length=max_length,
        padding="max_length",
        truncation="longest_first",
        return_tensors="pt",
    )
    if isinstance(le, LabelEncoder):
        encoded_y = le.transform(y)
    else:
        encoded_y = le.transform(np.array(y)[..., None]).astype(np.float32)

    labels = torch.from_numpy(encoded_y)
    return inputs, labels
