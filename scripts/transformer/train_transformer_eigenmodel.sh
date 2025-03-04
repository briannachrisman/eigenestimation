NNODES=1
N_PROC=4

WANDB_PROJECT="tinystories-8M-eigenmodel"
CHECKPOINT_PATH="outputs/eigenmodels/tinystories-8M.pt"

N_EIGENFEATURES=100
N_EIGENRANK=1
TOP_K=.3

EPOCHS=100
LEARNING_RATE=0.001
LR_STEP_EPOCHS=100
LR_DECAY_RATE=1
BATCH_SIZE=8
CHECKPOINT_EPOCHS=1
LOG_EPOCHS=1
TOKEN_LENGTH=16

#MODEL="roneneldan/TinyStories-1M"
#PARAMS="transformer.transformer.h.3.attn.attention.q_proj.weight,transformer.transformer.h.3.attn.attention.k_proj.weight,transformer.transformer.h.3.attn.attention.v_proj.weight"
#DATASET="roneneldan/TinyStories"
#TRAIN_SPLIT="train[:1%]"
#EVAL_SPLIT="validation[:1%]"


TOKENIZER="EleutherAI/gpt-neo-125M"


MODEL="roneneldan/TinyStories-8M"
PARAMS="transformer.transformer.h.1.attn.attention.q_proj.weight,transformer.transformer.h.1.attn.attention.k_proj.weight,transformer.transformer.h.1.attn.attention.v_proj.weight"

DATASET="roneneldan/TinyStories"
TRAIN_SPLIT="train[:1%]"
EVAL_SPLIT="validation[:1%]"
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
    --wandb-project $WANDB_PROJECT \
    --log-epochs $LOG_EPOCHS \
    --model $MODEL \
    --tokenizer $TOKENIZER \
    --params $PARAMS \
    --dataset $DATASET \
    --train-split $TRAIN_SPLIT \
    --eval-split $EVAL_SPLIT \
    --n-train-samples $N_TRAIN_SAMPLES \
    --n-eval-samples $N_EVAL_SAMPLES