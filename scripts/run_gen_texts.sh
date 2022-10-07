#!/usr/bin/env bash

args=(
    --run-id "ff7662944c014a1d9d774fd8e7edff8b"
    --n-samples-per-label 100
    --max-samples 2000000
    --top-k 50
    --top-p 0.95
)

python main.py gen-texts "${args[@]}"
