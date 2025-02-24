#!/bin/bash

NNODES=1
N_PROC=4

MODEL_PATH="outputs/toy_models/high_rank_circuits.pt"


N_EIGENFEATURES=10
TOP_K=0.1

SPARSITY=0.05
EPOCHS=1000
LEARNING_RATE=0.01
LR_STEP_EPOCHS=100
LR_DECAY_RATE=0.8
BATCH_SIZE=32
CHECKPOINT_EPOCHS=100
LOG_EPOCHS=10
N_TRAINING_DATAPOINTS=1000
N_EVAL_DATAPOINTS=100
CORRELATION_SET_SIZE=5



for N_EIGENRANK in 1 2 3 4 5 6 7 8; do

    WANDB_PROJECT="high-rank-circuits-eigenmodel-rank$N_EIGENRANK"
    CHECKPOINT_PATH="outputs/eigenmodels/high_rank_circuits_rank$N_EIGENRANK.pt"

    # Remove previous checkpoint
    if [ -f "$CHECKPOINT_PATH" ]; then
        echo "Removing previous checkpoint"
        rm -f "$CHECKPOINT_PATH"
    fi


    # Print directory script is in
    current_dir=$(echo "$(dirname "$(realpath "$0")")")
    # Run training
    torchrun --nnodes=$NNODES --nproc-per-node=$N_PROC $current_dir/train_high_rank_eigenmodel.py \
        --epochs $EPOCHS \
        --lr $LEARNING_RATE \
        --model-path $MODEL_PATH \
        --lr-step-epochs $LR_STEP_EPOCHS \
        --lr-decay-rate $LR_DECAY_RATE \
        --batch-size $BATCH_SIZE \
        --checkpoint-path $CHECKPOINT_PATH \
        --checkpoint-epochs $CHECKPOINT_EPOCHS \
        --n-eigenfeatures $N_EIGENFEATURES \
        --n-eigenrank $N_EIGENRANK \
        --top-k $TOP_K \
        --correlation-set-size $CORRELATION_SET_SIZE \
        --n-training-datapoints $N_TRAINING_DATAPOINTS \
        --n-eval-datapoints $N_EVAL_DATAPOINTS \
        --sparsity $SPARSITY \
        --wandb-project $WANDB_PROJECT \
        --log-epochs $LOG_EPOCHS

done

