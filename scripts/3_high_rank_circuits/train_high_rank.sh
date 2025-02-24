#!/bin/bash

NNODES=1
N_PROC=4  

WANDB_PROJECT="high-rank-circuits"
CHECKPOINT_PATH="/root/eigenestimation/outputs/toy_models/high_rank_circuits.pt"

N_FEATURES=30
N_HIDDEN=10
N_OUTPUTS=30

SPARSITY=0.05
EPOCHS=1000
LEARNING_RATE=0.001
BATCH_SIZE=32
CHECKPOINT_EPOCHS=100
LOG_EPOCHS=100
N_TRAINING_DATAPOINTS=10000
N_EVAL_DATAPOINTS=100
MAX_COEFFICIENT=3
CORRELATION_SET_SIZE=5
# Remove previous checkpoint
if [ -f "$CHECKPOINT_PATH" ]; then
    echo "Removing previous checkpoint"
    rm -f "$CHECKPOINT_PATH"
fi

# Print directory script is in
current_dir=$(echo "$(dirname "$(realpath "$0")")")
# Run training
torchrun --nnodes=$NNODES --nproc-per-node=$N_PROC $current_dir/train_high_rank.py \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --batch-size $BATCH_SIZE \
    --checkpoint-path $CHECKPOINT_PATH \
    --checkpoint-epochs $CHECKPOINT_EPOCHS \
    --n-features $N_FEATURES \
    --n-hidden $N_HIDDEN \
    --n-outputs $N_OUTPUTS \
    --n-training-datapoints $N_TRAINING_DATAPOINTS \
    --n-eval-datapoints $N_EVAL_DATAPOINTS \
    --sparsity $SPARSITY \
    --correlation-set-size $CORRELATION_SET_SIZE \
    --wandb-project $WANDB_PROJECT \
    --log-epochs $LOG_EPOCHS \
    --max-coefficient $MAX_COEFFICIENT

