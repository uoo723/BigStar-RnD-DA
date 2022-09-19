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
    --mp-enabled
    --run-id "6d26963c2d3f49d1a31439bd79b4996d"
)

python main.py train-baseline "${args[@]}"
