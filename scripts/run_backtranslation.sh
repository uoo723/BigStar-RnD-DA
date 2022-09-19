#!/usr/bin/env bash

args=(
    --batch-size 32
    --mp-enabled
)

python main.py backtranslation "${args[@]}"
