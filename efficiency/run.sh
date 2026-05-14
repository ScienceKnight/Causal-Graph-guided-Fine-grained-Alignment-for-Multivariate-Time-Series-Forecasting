#!/bin/bash

export CUDA_VISIBLE_DEVICES=0


for model in "bert-base-multilingual-cased" "bert-large-multilingual-cased" "xlm-roberta-base" "xlm-roberta-large"; do
    echo "Running benchmark for model: $model"
    python efficiency_benchmark.py --model_name $model > "benchmark_${model}.log" 2>&1
done

python aggregate_results.py