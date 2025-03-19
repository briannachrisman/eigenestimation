NNODES=1
N_PROC=1
CHECKPOINT_ARTIFACT_PATH="/root/eigenestimation/outputs/eigenmodels"

WANDB_PROJECT="mobilenet-v3-eigenmodel"

N_EIGENFEATURES=100
TOP_K=.1

EPOCHS=100
LEARNING_RATE=.005
LR_STEP_EPOCHS=100
LR_DECAY_RATE=.9
BATCH_SIZE=32
CHECKPOINT_EPOCHS=1
LOG_EPOCHS=1
CHUNK_SIZE=100
WARM_START_EPOCHS=10

CHECKPOINT_PATH="${CHECKPOINT_ARTIFACT_PATH}/${WANDB_PROJECT}.pt"

N_TRAIN_SAMPLES=10000
N_EVAL_SAMPLES=10

# Remove previous checkpoint
if [ -f "$CHECKPOINT_PATH" ]; then
    echo "Removing previous checkpoint"
    rm -f "$CHECKPOINT_PATH"
fi

# Print directory script is in
current_dir=$(echo "$(dirname "$(realpath "$0")")")
# Run training
torchrun --nnodes=$NNODES --nproc-per-node=$N_PROC $current_dir/train_cnn_eigenmodel.py \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --lr-step-epochs $LR_STEP_EPOCHS \
    --lr-decay-rate $LR_DECAY_RATE \
    --batch-size $BATCH_SIZE \
    --checkpoint-path $CHECKPOINT_PATH \
    --checkpoint-epochs $CHECKPOINT_EPOCHS \
    --n-eigenfeatures $N_EIGENFEATURES \
    --top-k $TOP_K \
    --wandb-project $WANDB_PROJECT \
    --log-epochs $LOG_EPOCHS \
    --chunk-size $CHUNK_SIZE \
    --n-train-samples $N_TRAIN_SAMPLES \
    --n-eval-samples $N_EVAL_SAMPLES \
    --warm-start-epochs $WARM_START_EPOCHS