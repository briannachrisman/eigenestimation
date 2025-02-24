#!/bin/bash

NNODES=1
N_PROC=4  

WANDB_PROJECT="tms-model"
CHECKPOINT_PATH="~/eigenestimation/outputs/toy_models/tms.pt"

N_FEATURES=5
N_HIDDEN=2
SPARSITY=0.05
EPOCHS=1000
LEARNING_RATE=0.001
BATCH_SIZE=32
CHECKPOINT_EPOCHS=500
LOG_EPOCHS=100
N_TRAINING_DATAPOINTS=10000
N_EVAL_DATAPOINTS=100


# Remove previous checkpoint
if [ -f "$CHECKPOINT_PATH" ]; then
    echo "Removing previous checkpoint"
    rm -f "$CHECKPOINT_PATH"
fi

# Print directory script is in
current_dir=$(echo "$(dirname "$(realpath "$0")")")
# Run training
torchrun --nnodes=$NNODES --nproc-per-node=$N_PROC $current_dir/train_tms.py \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --batch-size $BATCH_SIZE \
    --checkpoint-path $CHECKPOINT_PATH \
    --checkpoint-epochs $CHECKPOINT_EPOCHS \
    --n-features $N_FEATURES \
    --n-hidden $N_HIDDEN \
    --n-training-datapoints $N_TRAINING_DATAPOINTS \
    --n-eval-datapoints $N_EVAL_DATAPOINTS \
    --sparsity $SPARSITY \
    --wandb-project $WANDB_PROJECT \
    --log-epochs $LOG_EPOCHS
