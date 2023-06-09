"""
Created on 2022/06/07
@author Sangwoo Han
"""
import copy
import json
from ast import literal_eval
from functools import partial
from pathlib import Path
from typing import Any, Dict

import click
import optuna
from attrdict import AttrDict
from logzero import logger
from optuna import Study, Trial
from optuna.pruners import HyperbandPruner
from optuna.trial import TrialState
from ruamel.yaml import YAML

from main import cli
from src.utils import filter_arguments, log_elapsed_time
from train import train_model


def _load_train_params(config_filepath: str) -> AttrDict:
    with open(config_filepath, "r", encoding="utf-8") as f:
        return AttrDict(json.load(f))


def _suggest_value(trial: Trial, key: str, value: Dict[str, Any]) -> Any:
    if value["type"] == "categorical":
        suggest_func = trial.suggest_categorical
    elif value["type"] == "float":
        suggest_func = trial.suggest_float
    elif value["type"] == "int":
        suggest_func = trial.suggest_int
    elif value["type"] == "static":
        suggest_func = lambda name: value["value"]
    else:
        raise ValueError(f"{value['type']} is invalid")

    v = suggest_func(key, **filter_arguments(value, suggest_func))

    if "literal_eval" in value and value["literal_eval"]:
        v = literal_eval(v)

    if "round" in value:
        v = round(v, value["round"])

    return v


def _should_prune(cond_dict: Dict[str, Any]):
    if "prune" in cond_dict and cond_dict["prune"]:
        raise optuna.TrialPruned()


def _get_hp_params(trial: Trial, hp_params: Dict) -> Dict[str, Any]:
    p = {}
    for key, value in hp_params.items():
        p[key] = _suggest_value(trial, key, value)
        if "cond" in value:
            for cond in value["cond"]:
                if cond["cond_type"] == "eq" and p[key] == cond["cond_value"]:
                    _should_prune(cond)
                    p.update(_get_hp_params(trial, cond["cond_param"]))
                if cond["cond_type"] == "neq" and p[key] != cond["cond_value"]:
                    _should_prune(cond)
                    p.update(_get_hp_params(trial, cond["cond_param"]))
                elif cond["cond_type"] == "gt" and p[key] > cond["cond_value"]:
                    _should_prune(cond)
                    p.update(_get_hp_params(trial, cond["cond_param"]))
                elif cond["cond_type"] == "gte" and p[key] >= cond["cond_value"]:
                    _should_prune(cond)
                    p.update(_get_hp_params(trial, cond["cond_param"]))
                elif cond["cond_type"] == "lt" and p[key] < cond["cond_value"]:
                    _should_prune(cond)
                    p.update(_get_hp_params(trial, cond["cond_param"]))
                elif cond["cond_type"] == "lte" and p[key] <= cond["cond_value"]:
                    _should_prune(cond)
                    p.update(_get_hp_params(trial, cond["cond_param"]))
                elif cond["cond_type"] == "in" and p[key] in cond["cond_value"]:
                    _should_prune(cond)
                    p.update(_get_hp_params(trial, cond["cond_param"]))
                elif cond["cond_type"] == "nin" and p[key] not in cond["cond_value"]:
                    _should_prune(cond)
                    p.update(_get_hp_params(trial, cond["cond_param"]))
    return p


def _max_trial_callback(study: Study, trial: Trial, n_trials: int) -> None:
    n_complete = len(
        [
            t
            for t in study.trials
            if t.state in [TrialState.COMPLETE, TrialState.RUNNING, TrialState.PRUNED]
        ]
    )
    if n_complete >= n_trials:
        study.stop()


def objective(
    trial: Trial,
    train_params: Dict,
    hp_params: Dict,
    train_name: str,
    criterion: str,
    enable_trial_pruning: bool,
) -> float:
    params = copy.deepcopy(train_params)
    params.update(_get_hp_params(trial, hp_params))
    params.tags = list(params.tags) + [("trial", trial.number)]
    results = train_model(
        train_name,
        is_hptuning=True,
        trial=trial,
        enable_trial_pruning=enable_trial_pruning,
        **params,
    )
    return results[criterion]


@cli.command(context_settings={"show_default": True})
@click.option(
    "--hp-config-path",
    type=click.Path(exists=True),
    required=True,
    help="hp params config file path",
)
@click.option(
    "--train-config-path",
    type=click.Path(exists=True),
    required=True,
    help="train config file path",
)
@click.option("--n-trials", type=click.INT, default=20, help="# of trials")
@click.option("--study-name", type=click.STRING, default="study", help="Set study name")
@click.option(
    "--storage",
    type=click.STRING,
    default="sqlite:///./outputs/hpo_storage.db",
    help="Set storage path to save study",
)
@click.option(
    "--train-name",
    type=click.Choice(["attentionxml", "lightxml", "laroberta"]),
    default="attentionxml",
    help="Set train name",
)
@click.option(
    "--enable-trial-pruning",
    is_flag=True,
    default=False,
    help="enable trial pruning",
)
@log_elapsed_time
def hp_tuning(**args):
    """Hyper-parameter tuning"""
    args = AttrDict(args)

    yaml = YAML(typ="safe")
    hp_params = AttrDict(yaml.load(Path(args.hp_config_path)))
    train_params = _load_train_params(args.train_config_path)

    train_params.tags = [("study_name", args.study_name)]

    direction = "minimize" if train_params.early_criterion == "loss" else "maximize"

    if args.enable_trial_pruning:
        pruner = HyperbandPruner(
            min_resource=1, max_resource=train_params.num_epochs, reduction_factor=3
        )
    else:
        pruner = None

    study: Study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction=direction,
        pruner=pruner,
    )

    try:
        study.optimize(
            partial(
                objective,
                train_params=train_params,
                hp_params=hp_params,
                train_name=args.train_name,
                criterion="test/" + train_params.early_criterion,
                enable_trial_pruning=args.enable_trial_pruning,
            ),
            callbacks=[partial(_max_trial_callback, n_trials=args.n_trials)],
        )
    except KeyboardInterrupt:
        logger.info("Stop tuning.")

    all_trials = sorted(
        study.trials, key=lambda x: x.value if x.value else 0, reverse=True
    )
    best_trial = all_trials[0]

    best_exp_num = best_trial.number
    best_score = best_trial.value
    best_params = best_trial.params

    logger.info(f"best_exp_num: {best_exp_num}")
    logger.info(f"best_score: {best_score}")
    logger.info(f"best_params:\n{best_params}")
