"""
Created on 2022/09/10
@author Sangwoo Han
"""
import os
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class LotteQADataset(Dataset):
    def __init__(self, root_dir: str = "./data", mode: str = "train") -> None:
        assert mode in {"train", "valid", "test"}

        if mode == "train":
            df = pd.concat(
                [
                    pd.read_csv(
                        os.path.join(root_dir, "train01.csv"), low_memory=False
                    ),
                    pd.read_csv(
                        os.path.join(root_dir, "train02.csv"), low_memory=False
                    ),
                ]
            )
        elif mode == "valid":
            df = pd.read_csv(os.path.join(root_dir, "validation.csv"), low_memory=False)
        elif mode == "test":
            df = pd.read_csv(os.path.join(root_dir, "test.csv"), low_memory=False)

        self.df = df[df["QA여부"] == "q"]

    def __getitem__(self, index) -> Tuple[str, str]:
        return tuple(self.df[["발화문", "인텐트"]].iloc[index].tolist())

    def __len__(self) -> int:
        return len(self.df)

    @property
    def x(self) -> np.array:
        return self.df["발화문"].to_numpy()

    @property
    def y(self) -> np.array:
        return self.df["인텐트"].to_numpy()


def collate_fn(
    batch: Iterable[Tuple[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    le: LabelEncoder,
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
    labels = torch.from_numpy(le.transform(y))
    return inputs, labels
