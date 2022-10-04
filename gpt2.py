"""
Created on 2022/10/04
@author Sangwoo Han
"""
from pathlib import Path
from typing import Any, Set, Tuple, Union

import click
import pandas as pd
import torch
import torch.amp
from logzero import logger
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from main import cli
from src.base_trainer import get_ckpt_path, load_model_state
from src.utils import AttrDict, add_options, get_label_encoder, log_elapsed_time

# fmt: off

_options = [
    click.option("--no-cuda", is_flag=True, default=False, help="Disable cuda"),
    click.option("--pretrained-model-name", type=click.STRING, default="skt/kogpt2-base-v2", help="Pretrained model name"),
    click.option("--run-id", type=click.STRING, required=True, help="Run id which fine tunes GPT2"),
    click.option("--n-samples-per-label", type=click.INT, default=100, help="# of samples per label to be generated"),
    click.option("--max-samples", type=click.INT, default=10000, help="# of maximum samples to be generated"),
    click.option("--max-length", type=click.INT, default=100, help="The maximum length of the sequence to be generated"),
    click.option("--top-k", type=click.INT, help="Top k argument in GPT2LMHeadModel::generate"),
    click.option("--top-p", type=click.FLOAT, help="Top p argument in GPT2LMHeadModel::generate"),
    click.option("--save-interval", type=click.INT, default=500, help="Save interval"),
    click.option("--output-filepath", type=click.Path(), default="outputs/gpt2/aug_data.csv", help="Output filepath"),
]

# fmt: on


def _save_aug_data(aug_data: Set[Tuple[str, str]], filepath: Union[str, Path]) -> None:
    xs, ys = zip(*aug_data)
    df = pd.DataFrame({"발화문": xs, "인텐트": ys})
    df.to_csv(filepath, index=False)


@cli.command(context_settings={"show_default": True})
@add_options(_options)
@log_elapsed_time
def gen_texts(**args: Any) -> None:
    args = AttrDict(args)

    assert args.pretrained_model_name == "skt/kogpt2-base-v2"

    args.device = torch.device("cpu" if args.no_cuda else "cuda")
    args.output_filepath = Path(args.output_filepath)
    args.output_filepath.parent.mkdir(parents=True, exist_ok=True)

    le = get_label_encoder("./cache/label_encoder.joblib")

    logger.info("Load Model")
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        args.pretrained_model_name,
        bos_token="</s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name)
    model.to(args.device)

    ckpt_path = get_ckpt_path(log_dir="./logs", run_id=args.run_id)
    load_model_state(model, ckpt_path, substitution=(r"^model\.", ""))

    if args.output_filepath.exists():
        df = pd.read_csv(args.output_filepath)
        aug_data = set(zip(df["발화문"], df["인텐트"]))
    else:
        aug_data = set()

    label_token_map = {
        le.classes_[i]: f"<unused{i+1}>" for i in range(le.classes_.size)
    }
    sep_token = "<unused0>"

    step = 0
    with tqdm(total=args.max_samples) as pbar:
        pbar.update(len(aug_data))

        while len(aug_data) < args.max_samples:
            for label in le.classes_:
                text = [label_token_map[label] + sep_token]
                inputs = tokenizer(text, return_tensors="pt")
                output = model.generate(
                    inputs["input_ids"].to(args.device),
                    max_length=args.max_length,
                    do_sample=True,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    num_return_sequences=args.n_samples_per_label,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                )
                sentences = tokenizer.batch_decode(output, skip_special_tokens=True)
                num_aug_data = len(aug_data)
                aug_data.update(set(zip(sentences, [label] * len(sentences))))
                pbar.update(len(aug_data) - num_aug_data)

                step += 1

                if step % args.save_interval == 0:
                    _save_aug_data(aug_data, args.output_filepath)

    _save_aug_data(aug_data, args.output_filepath)
