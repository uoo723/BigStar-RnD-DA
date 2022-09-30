#!/usr/bin/env bash

args=(
    --batch-size 32
    --mp-enabled
    --over
    --output-filename "back_translated.v4.joblib"
    --save-interval 200
    --num-beams 5
    --early-stopping
)

python main.py backtranslation "${args[@]}"
