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
    --run-script $0
    --experiment-name $MLFLOW_EXPERIMENT_NAME
    --optim-name "adamw"
    --lr 5.5e-6
    --num-epochs 5
    --train-batch-size 32
    --test-batch-size 32
    --accumulation-step 1
    --scheduler-type "linear"
    --scheduler-warmup 0.05
    --early-criterion 'f1_weighted'
    --pretrained-model-name "klue/bert-base"
    --use-layernorm
    # --ls-alpha 0.1
    --max-length 100
    --seed $1
    --swa-warmup 0
    --eval-step 2000
    --early 10
    --mp-enabled
    --gradient-max-norm 5.0
    --num-workers 8
    --valid-size 1.0
    # --load-only-weights
    # --run-id "026111a273c14389885148f6a51a4582"
    --aug-filename "train+gpt2.v2.csv"
)

python main.py train-baseline "${args[@]}"
