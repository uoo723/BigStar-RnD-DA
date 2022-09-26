"""
Created on 2022/06/07
@author Sangwoo Han
"""
import copy
import inspect
import json
import os
import random
import shutil
import time
from datetime import timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import torch
from attrdict import AttrDict as _AttrDict
from logzero import logger
from sklearn.preprocessing import LabelEncoder


class AttrDict(_AttrDict):
    def _build(self, obj: Any) -> Any:
        return obj


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def filter_arguments(args: Dict[str, Any], obj: Any) -> Dict[str, Any]:
    if isinstance(obj, type):
        param_names = []
        for cls in inspect.getmro(obj):
            param_names.extend(list(inspect.signature(cls).parameters.keys()))
    else:
        param_names = list(inspect.signature(obj).parameters.keys())
    return {k: v for k, v in args.items() if k in param_names}


def log_elapsed_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()

        elapsed = end - start
        logger.info(f"elapsed time: {end - start:.2f}s, {timedelta(seconds=elapsed)}")

        return ret

    return wrapper


def save_args(args: AttrDict, path: str) -> None:
    args = copy.deepcopy(args)
    with open(path, "w", encoding="utf8") as f:
        json.dump(args, f, indent=4, ensure_ascii=False)


def load_args(path: str) -> AttrDict:
    with open(path, "r", encoding="utf8") as f:
        return AttrDict(json.load(f))


def copy_file(src: str, dst: str) -> None:
    try:
        shutil.copyfile(src, dst)
    except shutil.SameFileError:
        pass


def get_num_batches(batch_size: int, num_samples: int, drop_last: bool = False) -> int:
    if drop_last:
        return num_samples // batch_size
    return (num_samples + batch_size - 1) // batch_size


def get_label_encoder(
    path: Union[str, Path], y: Optional[np.array] = None
) -> LabelEncoder:
    if os.path.exists(path):
        return joblib.load(path)

    assert y is not None

    le = LabelEncoder().fit(y)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(le, path)

    return le


def get_n_samples(y: np.ndarray) -> torch.Tensor:
    _, n_samples = np.unique(y, return_counts=True)
    return torch.from_numpy(n_samples)


def delete_list_elements(list_: List[Any], indices: List[int]) -> None:
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_):
            list_.pop(idx)
