CHECKPOINT_ARTIFACT_PATH="/root/eigenestimation/outputs/eigenmodels"
NNODES=1
N_PROC=1

WANDB_PROJECT="tinystories-8M-eigenmodel"
N_EIGENFEATURES=100
TOP_K=.1

EPOCHS=100
LEARNING_RATE=0.001
LR_STEP_EPOCHS=100
LR_DECAY_RATE=.8
BATCH_SIZE=16
CHECKPOINT_EPOCHS=1
LOG_EPOCHS=1
TOKEN_LENGTH=16
CHUNK_SIZE=50
WARM_START_EPOCHS=10
TOKENIZER="EleutherAI/gpt-neo-125M"

MODEL="roneneldan/TinyStories-8M"

DATASET="roneneldan/TinyStories"
TRAIN_SPLIT="train[:1%]"
EVAL_SPLIT="validation[:1%]"
N_TRAIN_SAMPLES=10000
N_EVAL_SAMPLES=10
CHECKPOINT_PATH="${CHECKPOINT_ARTIFACT_PATH}/${WANDB_PROJECT}.pt"
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
    --warm-start-epochs $WARM_START_EPOCHS \
    --n-eigenfeatures $N_EIGENFEATURES \
    --top-k $TOP_K \
    --token-length $TOKEN_LENGTH \
    --wandb-project $WANDB_PROJECT \
    --log-epochs $LOG_EPOCHS \
    --model $MODEL \
    --tokenizer $TOKENIZER \
    --dataset $DATASET \
    --train-split $TRAIN_SPLIT \
    --eval-split $EVAL_SPLIT \
    --chunk-size $CHUNK_SIZE \
    --n-train-samples $N_TRAIN_SAMPLES \
    --n-eval-samples $N_EVAL_SAMPLES