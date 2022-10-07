#!/usr/bin/env bash
# export MLFLOW_TRACKING_URI=http://localhost:5000
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MLFLOW_EXPERIMENT_NAME=Baseline2
export TRANSFORMERS_VERBOSITY=error

DATASET=LotteQA
MODEL=Baseline

args=(
    --model-name $MODEL
    --dataset-name $DATASET
    --experiment-name $MLFLOW_EXPERIMENT_NAME
    --run-script $0
    --seed $1
    --optim-name "adamw"
    --lr 5.5e-6
    --num-epochs 50
    --train-batch-size 32
    --test-batch-size 32
    --accumulation-step 1
    --scheduler-type "linear"
    --scheduler-warmup 0.1
    --early-criterion 'f1_weighted'
    --pretrained-model-name "klue/bert-base"
    --use-layernorm
    # --ls-alpha 0.1
    --max-length 100
    --swa-warmup 0
    --eval-step 2000
    --early 10
    --mp-enabled
    --gradient-max-norm 5.0
    --num-workers 8
    --valid-size 1.0
    # --load-only-weights
    # --run-id "f3043d2eda5349ee99b452640bca1a35"
    # --aug-filename "train_samplingv5+gpt2_datav4.csv"
)

python main.py train-baseline "${args[@]}"
