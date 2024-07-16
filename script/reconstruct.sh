#!/bin/bash

SUBJ_ID=${1}
TRAIN_MODE=${2}
USE_VIS_MASK=${3}
SEED=${4}
RECON_PER_SAMPLE=${5}
AVG_DATA_DIR="data/WAVE-BOLD5000"
CHECKPOINT_PRIOR_DIR='checkpoints/prior'
CHEKPOINT_SD_DIR='checkpoints/sd-image-variations-diffusers'
CHECKPOINT_VD_DIR='checkpoints/vd'
CHECKPOINT_TRAINED_DIR=checkpoints/WAVE-models/$SUBJ_ID/decode/model_last_prompt_vd.bin
CHECKPOINT_AUTOENC_DIR=checkpoints/WAVE-models/$SUBJ_ID/autoenc/model_last_mindeye_bold5000_voxel.pth
VIS_MASK_JSON=None
if [ $USE_VIS_MASK == 'True' ]; then
        VIS_MASK_JSON='checkpoints/WAVE-models/index_yeo7.json'
        CHECKPOINT_TRAINED_DIR=checkpoints/WAVE-models/$SUBJ_ID/decode/model_last_prompt_vd_vis-mask.bin
fi
echo "The trained checkpoint in use is: $CHECKPOINT_TRAINED_DIR"
TRAINING_STYLE='CSM'
LR=1e-5
LOG_DIR='results/models/BOLD5000/reconstruct-fmri/'

conda activate wave
cd ..

python src/reconstruct.py --architecture 'GPT' \
        --training-style $TRAINING_STYLE \
        --training-steps 400 \
        --validation-steps 30 \
        --per-device-training-batch-size 16 \
        --per-device-validation-batch-size 16 \
        --learning-rate $LR \
        --log-dir $LOG_DIR \
        --log-every-n-steps 100 \
        --is-prompt False \
        --wandb True \
        --l1-lambda 0 \
        --scheduler 'step' \
        --fp16 True \
        --set-seed True \
        --seed $SEED \
        --sparse-ratio 0 \
        --scheduler-step-size 1000 \
        --scheduler-gamma 1 \
        --weight-decay 0.02 \
        --eval-every-n-steps 1000 \
        --subject-id $SUBJ_ID \
        --only-visual True \
        --prior-checkpoint-dir $CHECKPOINT_PRIOR_DIR \
        --sd-ckpt-dir $CHEKPOINT_SD_DIR \
        --n-samples-save 1 \
        --log-reconstruction-every-n-steps 1000 \
        --hidden True \
        --trained-checkpoint-dir $CHECKPOINT_TRAINED_DIR \
        --save-recon-embds False \
        --vd-ckpt-dir $CHECKPOINT_VD_DIR \
        --autoenc-ckpt-dir $CHECKPOINT_AUTOENC_DIR \
        --avg-data-dir $AVG_DATA_DIR \
        --recons-per-sample $RECON_PER_SAMPLE \
        --vis-mask-json $VIS_MASK_JSON \
        --train-mode $TRAIN_MODE 
        
cd script
conda deactivate