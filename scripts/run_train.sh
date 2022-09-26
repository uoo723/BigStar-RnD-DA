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
    --lr 5e-5
    --num-epochs 10
    --train-batch-size 32
    --test-batch-size 32
    --accumulation-step 1
    --scheduler-type "linear"
    --scheduler-warmup 0.02
    --early-criterion 'f1_micro'
    --pretrained-model-name "klue/bert-base"
    --use-layernorm
    --max-length 100
    --seed $1
    --swa-warmup 0
    --eval-step 2000
    --early 5
    --mp-enabled
    --gradient-max-norm 5.0
    --num-workers 8
    --experiment-name "Baseline"
    --valid-size 1.0
    # --load-only-weights
    # --run-id "8119d5093bb5486ea9c23f7954755deb"
    --aug-filename "train+back.v7.csv"
)

python main.py train-baseline "${args[@]}"
