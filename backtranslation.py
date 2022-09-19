"""
Created on 2022/09/19
@author Sangwoo Han
"""
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import click
import joblib
import torch
import torch.amp
from logzero import logger
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, MarianMTModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from main import cli
from src.datasets import LotteQADataset
from src.utils import AttrDict, add_options, log_elapsed_time

# fmt: off

_options = [
    click.option("--batch-size", type=click.INT, default=32, help="Batch size"),
    click.option("--no-cuda", is_flag=True, default=False, help="Disable cuda"),
    click.option("--mp-enabled", is_flag=True, default=False, help="Enable Mixed Precision"),
    click.option("--num-workers", type=click.INT, default=4, help="Number of workers for data loader"),
    click.option("--output-dir", type=click.Path(), default="./outputs/backtranslation", help="Output path"),
]

# fmt: on


class ListDataset(Dataset):
    def __init__(self, data: List[str]) -> None:
        self.data = data

    def __getitem__(self, index: int) -> str:
        return (self.data[index],)

    def __len__(self) -> int:
        return len(self.data)


def _collate_fn_seq2seq(
    batch: Iterable[Tuple[str, str]],
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[Dict[str, torch.Tensor]]:
    x = [b[0] for b in batch]
    inputs = tokenizer(x, padding=True, return_tensors="pt")
    return inputs


def _generate_seq(
    batch_x: Dict[str, Any],
    model: MarianMTModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device = torch.device("cpu"),
    mp_enabled: bool = False,
) -> List[str]:
    model.eval()
    batch_x = {k: v.to(device) for k, v in batch_x.items()}
    with torch.no_grad(), torch.amp.autocast(str(device), enabled=mp_enabled):
        encoded = model.generate(**batch_x)

    with tokenizer.as_target_tokenizer():
        return tokenizer.batch_decode(encoded, skip_special_tokens=True)


@cli.command(context_settings={"show_default": True})
@add_options(_options)
@log_elapsed_time
def backtranslation(**args: Any) -> None:
    args = AttrDict(args)
    device = torch.device("cpu" if args.no_cuda else "cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    logger.info("Load Dataset")
    train_dataset = LotteQADataset()

    logger.info(f"# of train: {len(train_dataset):,}")

    logger.info("Load src model")
    src_model_name = "Helsinki-NLP/opus-mt-ko-en"
    src_model = MarianMTModel.from_pretrained(src_model_name)
    src_model.to(device)
    src_tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        src_model_name
    )

    logger.info("Load tgt model")
    tgt_model_name = "Helsinki-NLP/opus-mt-tc-big-en-ko"
    tgt_model = MarianMTModel.from_pretrained(tgt_model_name)
    tgt_model.to(device)
    tgt_tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        "en_ko_tokenizer"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=partial(_collate_fn_seq2seq, tokenizer=src_tokenizer),
        pin_memory=not args.no_cuda,
        num_workers=args.num_workers,
    )

    src_translated = []
    for batch_x in tqdm(train_dataloader, desc="ko..en"):
        src_translated.extend(
            _generate_seq(batch_x, src_model, src_tokenizer, device, args.mp_enabled)
        )

    joblib.dump(src_translated, output_dir / "src_translated.joblib")

    train_dataloader = DataLoader(
        ListDataset(src_translated),
        batch_size=args.batch_size,
        collate_fn=partial(_collate_fn_seq2seq, tokenizer=tgt_tokenizer),
        pin_memory=not args.no_cuda,
        num_workers=args.num_workers,
    )

    tgt_translated = []
    for batch_x in tqdm(train_dataloader, desc="en..ko"):
        tgt_translated.extend(
            _generate_seq(batch_x, tgt_model, tgt_tokenizer, device, args.mp_enabled)
        )

    joblib.dump(tgt_translated, output_dir / "tgt_translated.joblib")
