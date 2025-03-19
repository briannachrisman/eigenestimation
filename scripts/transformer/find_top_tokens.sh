NNODES=1
N_PROC=1
CHECKPOINT_ARTIFACT_PATH="/root/eigenestimation/outputs/eigenmodels"

EIGENMODEL_PATH="$CHECKPOINT_ARTIFACT_PATH/tinystories-8M-eigenmodel.pt"
DATASET="roneneldan/TinyStories"
SPLIT="validation[:1%]"
N_SAMPLES=10000
PAIRED_ITERS=10
BATCH_SIZE=16
ATT_OUTPUT_FILE="$CHECKPOINT_ARTIFACT_PATH/tinystories-8M-circuit_attributions.pt"
EXAMPLES_OUTPUT_FILE="$CHECKPOINT_ARTIFACT_PATH/tinystories-8M-X_data.pt"
TOKEN_LENGTH=16
TOP_K=5
JAC_CHUNK_SIZE=25
current_dir=$(echo "$(dirname "$(realpath "$0")")")
echo "current dir: $current_dir"
# Run training
torchrun --nnodes=$NNODES --nproc-per-node=$N_PROC $current_dir/find_top_tokens.py \
--model-path $EIGENMODEL_PATH \
--dataset $DATASET \
--split $SPLIT \
--token-length $TOKEN_LENGTH \
--num-samples $N_SAMPLES \
--iters $PAIRED_ITERS \
--batch-size $BATCH_SIZE \
--top-k $TOP_K \
--attributions-output-file $ATT_OUTPUT_FILE \
--examples-output-file $EXAMPLES_OUTPUT_FILE \
--jac-chunk-size $JAC_CHUNK_SIZE