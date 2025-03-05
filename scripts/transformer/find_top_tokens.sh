NNODES=1
N_PROC=1

EIGENMODEL_PATH="/root/eigenestimation/outputs/eigenmodels/tinystories-8M.pt"
DATASET="roneneldan/TinyStories"
SPLIT="validation[:1000]"
N_SAMPLES=100
PAIRED_ITERS=5
BATCH_SIZE=32
ATT_OUTPUT_FILE="/root/eigenestimation/outputs/top_tokens/tinystories-8M-circuit_attributions.pt"
EXAMPLES_OUTPUT_FILE="/root/eigenestimation/outputs/top_tokens/tinystories-8M-X_data.pt"
TOKEN_LENGTH=16
TOP_K=5
JAC_CHUNK_SIZE=50
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