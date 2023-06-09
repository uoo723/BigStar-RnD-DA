"""
Created on 2022/09/19
@author Sangwoo Han
"""
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import click
import joblib
import numpy as np
import torch
import torch.amp
from logzero import logger
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, MarianMTModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from main import cli
from src.datasets import LotteQADataset
from src.utils import (
    AttrDict,
    add_options,
    delete_list_elements,
    get_label_encoder,
    get_n_samples,
    log_elapsed_time,
)

# fmt: off

_options = [
    click.option("--batch-size", type=click.INT, default=32, help="Batch size"),
    click.option("--no-cuda", is_flag=True, default=False, help="Disable cuda"),
    click.option("--mp-enabled", is_flag=True, default=False, help="Enable Mixed Precision"),
    click.option("--num-workers", type=click.INT, default=4, help="Number of workers for data loader"),
    click.option("--output-dir", type=click.Path(), default="./outputs/backtranslation", help="Output path"),
    click.option("--output-filename", type=click.STRING, default="back_translated.joblib", help="Output filename"),
    click.option("--aug-filename", type=click.STRING, help="filename in which data is augmented"),
    click.option("--cache-dir", type=click.Path(), default="./cache", help="Cache directory"),
    click.option("--over", is_flag=True, default=False, help="Use over sampling"),
    click.option("--max-samples", type=click.INT, help="Max # of generated samples"),
    click.option("--save-interval", type=click.INT, default=500, help="Save interval"),
    click.option("--num-beams", type=click.INT, default=1, help="# of beams for beam search"),
    click.option("--do-sample", is_flag=True, default=False, help="Whether or not to use sampling"),
    click.option("--early-stopping", is_flag=True, default=False, help="Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not"),
    click.option("--max-length", type=click.INT, help="The maximum length of the sequence to be generated"),
]

# fmt: on


class ListDataset(Dataset):
    def __init__(self, x: List[str], y: List[str]) -> None:
        self.x = x
        self.y = y

    def __getitem__(self, index: int) -> str:
        return (self.x[index], self.y[index])

    def __len__(self) -> int:
        return len(self.x)


def _collate_fn_seq2seq(
    batch: Iterable[Tuple[str, str]]
) -> Tuple[List[str], List[str]]:
    x = [b[0] for b in batch]
    y = [b[1] for b in batch]
    return x, y


def _generate_seq(
    batch_x: Dict[str, Any],
    model: MarianMTModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device = torch.device("cpu"),
    mp_enabled: bool = False,
    num_beams: int = 1,
    do_sample: bool = False,
    early_stopping: bool = False,
    max_length: Optional[int] = None,
) -> List[str]:
    model.eval()
    batch_x = {k: v.to(device) for k, v in batch_x.items()}
    with torch.amp.autocast(device.type, enabled=mp_enabled):
        encoded = model.generate(
            batch_x["input_ids"],
            num_beams=num_beams,
            do_sample=do_sample,
            early_stopping=early_stopping,
            max_length=max_length,
        )

    with tokenizer.as_target_tokenizer():
        return tokenizer.batch_decode(encoded, skip_special_tokens=True)


def _backtranslate(
    sents: List[str],
    src_model: MarianMTModel,
    src_tokenizer: PreTrainedTokenizerBase,
    tgt_model: MarianMTModel,
    tgt_tokenizer: PreTrainedTokenizerBase,
    device: torch.device = torch.device("cpu"),
    mp_enabled: bool = False,
    **kwargs: Any,
) -> List[str]:
    # Translate from source language to target language
    inputs = src_tokenizer(
        sents, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    translated = _generate_seq(
        inputs, src_model, src_tokenizer, device, mp_enabled, **kwargs
    )

    # Translate from target language back to source language
    inputs = tgt_tokenizer(
        translated, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    back_translated = _generate_seq(
        inputs, tgt_model, tgt_tokenizer, device, mp_enabled, **kwargs
    )

    return back_translated


def _generate_samples(
    dataset: LotteQADataset,
    src_model: MarianMTModel,
    src_tokenizer: PreTrainedTokenizerBase,
    tgt_model: MarianMTModel,
    tgt_tokenizer: PreTrainedTokenizerBase,
    args: AttrDict,
) -> None:
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=_collate_fn_seq2seq,
        pin_memory=not args.no_cuda,
        num_workers=args.num_workers,
    )

    back_translated = []
    for batch_x, batch_y in tqdm(dataloader):
        back_translated.extend(
            list(
                zip(
                    _backtranslate(
                        batch_x,
                        src_model,
                        src_tokenizer,
                        tgt_model,
                        tgt_tokenizer,
                        args.device,
                        args.mp_enabled,
                    ),
                    batch_y,
                )
            )
        )

    joblib.dump(back_translated, args.output_dir / args.output_filename)


def _generate_samples_with_over(
    dataset: LotteQADataset,
    src_model: MarianMTModel,
    src_tokenizer: PreTrainedTokenizerBase,
    tgt_model: MarianMTModel,
    tgt_tokenizer: PreTrainedTokenizerBase,
    args: AttrDict,
) -> None:
    le = get_label_encoder(args.cache_dir / "label_encoder.joblib", dataset.y)
    output_path: Path = args.output_dir / args.output_filename
    cache_dir: Path = (
        args.cache_dir / hashlib.md5(str(output_path).encode("utf8")).hexdigest()
    )

    cache_dir.mkdir(parents=True, exist_ok=True)

    xs, ys = [], []

    if output_path.exists():
        back_translated = set(joblib.load(output_path))
        data = joblib.load(cache_dir / "data.joblib")
        xs, ys = data["xs"], data["ys"]
    else:
        back_translated = set()

    if len(xs) == 0:
        xs, ys = dataset.x.tolist(), dataset.y.tolist()

    n_samples = get_n_samples(le.transform(ys))

    num_steps = 0
    max_samples = args.max_samples or len(dataset)
    batch_data = set()
    with tqdm(total=max_samples) as pbar:
        pbar.update(len(back_translated))
        while len(back_translated) < max_samples:
            batch_idx = np.random.choice(len(xs), size=args.batch_size, replace=False)
            batch_x = np.array([xs[i] for i in batch_idx])
            batch_y = le.transform([ys[i] for i in batch_idx])
            probs = (1 - n_samples[batch_y] / n_samples.max()) + 1e-4
            idx = probs.clamp(max=1.0).bernoulli().nonzero(as_tuple=True)[0].reshape(-1)

            if idx.nelement() == 0:
                continue

            idx = idx.numpy()
            batch_data.update(
                set(
                    zip(
                        batch_idx[idx].tolist(),
                        batch_x[idx].tolist(),
                        le.classes_[batch_y[idx]].tolist(),
                    )
                )
            )

            if len(batch_data) < args.batch_size:
                continue

            batch_idx, batch_x, batch_y = zip(*batch_data)
            batch_idx, batch_x, batch_y = list(batch_idx), list(batch_x), list(batch_y)
            aug_data = (
                set(
                    zip(
                        _backtranslate(
                            batch_x[: args.batch_size],
                            src_model,
                            src_tokenizer,
                            tgt_model,
                            tgt_tokenizer,
                            args.device,
                            args.mp_enabled,
                            num_beams=args.num_beams,
                            do_sample=args.do_sample,
                            early_stopping=args.early_stopping,
                            max_length=args.max_length,
                        ),
                        batch_y[: args.batch_size],
                    )
                )
                - back_translated
            )

            delete_list_elements(xs, batch_idx[: args.batch_size])
            delete_list_elements(ys, batch_idx[: args.batch_size])
            assert len(xs) == len(ys)

            batch_data.clear()

            if not aug_data:
                continue

            aug_data_size = min(args.batch_size, max_samples - len(back_translated))
            aug_data = set(list(aug_data)[:aug_data_size])
            back_translated.update(aug_data)
            pbar.update(len(aug_data))

            aug_texts, aug_labels = zip(*aug_data)
            xs.extend(aug_texts)
            ys.extend(aug_labels)

            for l in le.transform(aug_labels):
                n_samples[l] += 1

            num_steps += 1

            if num_steps % args.save_interval == 0:
                joblib.dump(list(back_translated), output_path)
                joblib.dump({"xs": xs, "ys": ys}, cache_dir / "data.joblib")

    joblib.dump(list(back_translated), output_path)
    joblib.dump({"xs": xs, "ys": ys}, cache_dir / "data.joblib")


@cli.command(context_settings={"show_default": True})
@add_options(_options)
@log_elapsed_time
def backtranslation(**args: Any) -> None:
    args = AttrDict(args)
    args.device = torch.device("cpu" if args.no_cuda else "cuda")
    args.output_dir = Path(args.output_dir)
    args.cache_dir = Path(args.cache_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Load Dataset")
    train_dataset = LotteQADataset(aug_filename=args.aug_filename)

    logger.info(f"# of train: {len(train_dataset):,}")

    logger.info("Load src model")
    src_model_name = "Helsinki-NLP/opus-mt-ko-en"
    src_model = MarianMTModel.from_pretrained(src_model_name)
    src_model.to(args.device)
    src_tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        src_model_name
    )

    logger.info("Load tgt model")
    tgt_model_name = "Helsinki-NLP/opus-mt-tc-big-en-ko"
    tgt_model = MarianMTModel.from_pretrained(tgt_model_name)
    tgt_model.to(args.device)
    tgt_tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        "en_ko_tokenizer"
    )

    if not args.over:
        _generate_samples(
            train_dataset, src_model, src_tokenizer, tgt_model, tgt_tokenizer, args
        )
    else:
        _generate_samples_with_over(
            train_dataset, src_model, src_tokenizer, tgt_model, tgt_tokenizer, args
        )
