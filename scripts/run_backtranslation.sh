#!/usr/bin/env bash

args=(
    --batch-size 32
    --mp-enabled
    --over
)

python main.py backtranslation "${args[@]}"
