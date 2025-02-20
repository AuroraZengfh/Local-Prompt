#!/bin/bash
# custom config
TRAINER=LOCALPROMPT

DATA=$1
DATASET=$2
CFG=$3  # config file
CTP=$4   # class token position (end or middle)
NCTX=$5  # number of context tokens
SHOTS=$6   # number of shots (1, 2, 4, 8, 16)
CSC=$7  # class-specific context (False or True)
lambda=$8
div_value=$9
topk=$10

for SEED in 1 2 3
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}_topk${topk}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        echo $PWD
        CUDA_VISIBLE_DEVICES=0 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --lambda_value ${lambda} \
        --div_value ${div_value} \
        --topk ${topk} \
        TRAINER.${TRAINER}.N_CTX ${NCTX} \
        TRAINER.${TRAINER}.CSC ${CSC} \
        TRAINER.${TRAINER}.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done