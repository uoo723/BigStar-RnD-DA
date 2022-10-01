#!/usr/bin/env bash
# export MLFLOW_TRACKING_URI=http://localhost:5000
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MLFLOW_EXPERIMENT_NAME=GPT2
export TRANSFORMERS_VERBOSITY=error

DATASET=LotteQA
MODEL=GPT2

args=(
    --model-name $MODEL
    --dataset-name $DATASET
    --experiment-name $MLFLOW_EXPERIMENT_NAME
    --run-script $0
    --skip-test
    --optim-name "adamw"
    --lr 5e-5
    --num-epochs 1
    --train-batch-size 2
    --block-size 1024
    --accumulation-step 1
    --scheduler-type "linear"
    --scheduler-warmup 0.1
    --seed $1
    --swa-warmup 0
    --eval-step 1
    --early 0
    --mp-enabled
    --gradient-max-norm 5.0
    --num-workers 8
)

python main.py train-gpt2 "${args[@]}"
