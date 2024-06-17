#!/bin/bash

SUBJ_ID=${1}
USE_VIS_MASK=${2}
AVG_DATA_DIR="data/BOLD5000"
CHECKPOINT_DIR=checkpoints/$SUBJ_ID/contrastive/model_best_prompt.bin
CHECKPOINT_PRIOR_DIR='/scratch/yl6624/fmri-ecg/checkpoints/fmri/prior'
CHECKPOINT_VD_DIR='checkpoints/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7'
CHEKPOINT_SD_DIR='checkpoints/sd-image-variations-diffusers'
TRAINING_STYLE='CSM'
VIS_MASK_JSON=None
if [ $USE_VIS_MASK == 'True' ]; then
    VIS_MASK_JSON='checkpoints/ndex_yeo7.json'
    CHECKPOINT_DIR=checkpoints/$SUBJ_ID/contrastive/model_best_prompt_vis-mask.bin
fi

TARFILE_JSON='/scratch/yl6624/fmri-ecg/config/tarfile/ds_bold5000_subj_3_no_rep.json'
LR=1e-5
LOG_DIR='results/models/BOLD5000/decode-fmri/'

conda activate wave
cd ..

python src/train_bold5000_decode.py --architecture 'GPT' \
        --pretrained-model $CHECKPOINT_DIR \
        --training-style $TRAINING_STYLE \
        --training-steps 200000 \
        --validation-steps 30 \
        --per-device-training-batch-size 16 \
        --per-device-validation-batch-size 16 \
        --learning-rate $LR \
        --log-dir $LOG_DIR \
        --log-every-n-steps 100 \
        --is-prompt False \
        --wandb False \
        --l1-lambda 0 \
        --scheduler 'step' \
        --fp16 True \
        --set-seed True \
        --seed 42 \
        --sparse-ratio 0 \
        --scheduler-step-size 1000 \
        --scheduler-gamma 1 \
        --weight-decay 0.02 \
        --eval-every-n-steps 1000 \
        --subject-id $SUBJ_ID \
        --train-perc 0.9 \
        --eval-perc 0.1 \
        --test-perc 0 \
        --only-visual True \
        --prior-checkpoint-dir $CHECKPOINT_PRIOR_DIR \
        --sd-ckpt-dir $CHEKPOINT_SD_DIR \
        --n-samples-save 1 \
        --log-reconstruction-every-n-steps 1000 \
        --tarfile-paths-split $TARFILE_JSON \
        --hidden True \
        --vd-ckpt-dir $CHECKPOINT_VD_DIR \
        --train-mode 'region' \
        --vis-mask-json $VIS_MASK_JSON \
        --avg-data-dir $AVG_DATA_DIR
cd script
# conda deactivate