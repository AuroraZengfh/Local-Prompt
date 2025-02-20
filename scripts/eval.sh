#!/bin/bash

# custom config
TRAINER=LOCALPROMPT

CSC=True
CTP=end

DATA=$1
DATASET=$2
CFG=$3
NCTX=$4
topk=$5

MODEL_dir=$6
Output_dir=$7

CUDA_VISIBLE_DEVICES=0 python eval_ood_detection.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--in_dataset ${DATASET} \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--load-epoch 30 \
--output-dir ${Output_dir} \
--model-dir ${MODEL_dir} \
--top_k ${topk} \
TRAINER.LOCALPROMPT.N_CTX ${NCTX} \
TRAINER.LOCALPROMPT.CSC ${CSC} \
