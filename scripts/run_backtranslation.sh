#!/usr/bin/env bash

args=(
    --batch-size 32
    --mp-enabled
    --over
    --output-filename "back_translated.v2.joblib"
)

python main.py backtranslation "${args[@]}"
