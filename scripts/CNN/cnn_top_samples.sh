NNODES=1
N_PROC=1
CHECKPOINT_ARTIFACT_PATH="/root/eigenestimation/outputs/eigenmodels"
EIGENMODEL_PATH="$CHECKPOINT_ARTIFACT_PATH/mobilenet-v3-eigenmodel.pt"
N_SAMPLES=10000
PAIRED_ITERS=10
BATCH_SIZE=64
ATT_OUTPUT_FILE="$CHECKPOINT_ARTIFACT_PATH/mobilenet-v3-eigenmodel-circuit_attributions.pt"
EXAMPLES_OUTPUT_FILE="$CHECKPOINT_ARTIFACT_PATH/mobilenet-v3-eigenmodel-X_data.pt"
JAC_CHUNK_SIZE=1000
current_dir=$(echo "$(dirname "$(realpath "$0")")")
echo "current dir: $current_dir"
# Run training
torchrun --nnodes=$NNODES --nproc-per-node=$N_PROC $current_dir/cnn_top_samples.py \
--model-path $EIGENMODEL_PATH \
--num-samples $N_SAMPLES \
--iters $PAIRED_ITERS \
--batch-size $BATCH_SIZE \
--attributions-output-file $ATT_OUTPUT_FILE \
--examples-output-file $EXAMPLES_OUTPUT_FILE \
--jac-chunk-size $JAC_CHUNK_SIZE