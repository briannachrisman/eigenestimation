#!/bin/bash

NNODES=1
N_PROC=1

WANDB_PROJECT="circuit-superposition-eigenmodel"
MODEL_PATH="outputs/toy_models/circuit_superposition.pt"
CHECKPOINT_PATH="outputs/eigenmodels/circuit_superposition.pt"

N_EIGENFEATURES=10
N_EIGENRANK=1
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
N_EVAL_DATAPOINTS=1000
CHUNK_SIZE=100

# Remove previous checkpoint
if [ -f "$CHECKPOINT_PATH" ]; then
    echo "Removing previous checkpoint"
    rm -f "$CHECKPOINT_PATH"
fi

# Print directory script is in
current_dir=$(echo "$(dirname "$(realpath "$0")")")
# Run training
torchrun --nnodes=$NNODES --nproc-per-node=$N_PROC $current_dir/train_circuit_superposition_eigenmodel.py \
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
    --n-training-datapoints $N_TRAINING_DATAPOINTS \
    --n-eval-datapoints $N_EVAL_DATAPOINTS \
    --sparsity $SPARSITY \
    --wandb-project $WANDB_PROJECT \
    --chunk-size $CHUNK_SIZE \
    --warm-start-epochs 1 \
    --log-epochs $LOG_EPOCHS

