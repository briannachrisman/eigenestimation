NNODES=1
N_PROC=4

WANDB_PROJECT="transformer-eigenmodel"
CHECKPOINT_PATH="outputs/eigenmodels/transformer.pt"

N_EIGENFEATURES=30
N_EIGENRANK=3
TOP_K=0.1

EPOCHS=1000
LEARNING_RATE=0.001
LR_STEP_EPOCHS=100
LR_DECAY_RATE=0.8
BATCH_SIZE=8
CHECKPOINT_EPOCHS=100
LOG_EPOCHS=10
N_TRAINING_DATAPOINTS=1000
N_EVAL_DATAPOINTS=100
TOKEN_LENGTH=16

# Remove previous checkpoint
if [ -f "$CHECKPOINT_PATH" ]; then
    echo "Removing previous checkpoint"
    rm -f "$CHECKPOINT_PATH"
fi

# Print directory script is in
current_dir=$(echo "$(dirname "$(realpath "$0")")")
# Run training
torchrun --nnodes=$NNODES --nproc-per-node=$N_PROC $current_dir/train_transformer_eigenmodel.py \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --lr-step-epochs $LR_STEP_EPOCHS \
    --lr-decay-rate $LR_DECAY_RATE \
    --batch-size $BATCH_SIZE \
    --checkpoint-path $CHECKPOINT_PATH \
    --checkpoint-epochs $CHECKPOINT_EPOCHS \
    --n-eigenfeatures $N_EIGENFEATURES \
    --n-eigenrank $N_EIGENRANK \
    --top-k $TOP_K \
    --token-length $TOKEN_LENGTH \
    --n-training-datapoints $N_TRAINING_DATAPOINTS \
    --n-eval-datapoints $N_EVAL_DATAPOINTS \
    --wandb-project $WANDB_PROJECT \
    --log-epochs $LOG_EPOCHS