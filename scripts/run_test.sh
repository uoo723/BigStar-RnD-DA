#!/usr/bin/env bash
# export MLFLOW_TRACKING_URI=http://localhost:5000
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# export MLFLOW_EXPERIMENT_NAME=Baseline

DATASET=LotteQA
MODEL=Baseline

args=(
    --model-name $MODEL
    --dataset-name $DATASET
    --run-script $0
    --optim-name "adamw"
    --mode "test"
    --test-batch-size 64
    --run-id "8119d5093bb5486ea9c23f7954755deb"
)

python main.py train-baseline "${args[@]}"
