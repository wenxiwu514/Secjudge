#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python main.py \
    --api hf \
    --dataset sst5 \
    --data_file data/sst5/sst5_test.csv \
    --word_embedding_model distilbert-base-uncased \
    --phrase_model Qwen/Qwen2.5-1.5B-Instruct \
    --combine_divide 4 \
    --epochs 1 \
    --num_private_samples 10 \
    --result_folder result \
    --feature_extractor_batch_size 1024 \
    --feature_extractor all-mpnet-base-v2 \
    --noise_multiplier 0 \
    --nn_mode L2 \
    --count_threshold 0.0 \
    --select_syn_mode rank \
    --save_syn_mode selected \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --variation_batch_size 128 \
    --length 128