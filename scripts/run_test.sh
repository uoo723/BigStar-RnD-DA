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
    --test-batch-size 32
    --run-id "201d41530e1c45938830e23191293e4c"
)

python main.py train-baseline "${args[@]}"
