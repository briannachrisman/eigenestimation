source ~/.eigenestimation/bin/activate
cd ~/eigenestimation/
NNODES=1
N_PROC=4
for i in {1..4}; do
    if [ -f "outputs/eigenmodels/high_rank_eigenmodel_rank${i}.pt" ]; then
        rm outputs/eigenmodels/high_rank_eigenmodel_rank${i}.pt
    fi
    torchrun  --nnodes=$NNODES --nproc-per-node=$N_PROC  scripts/3_high_rank_circuits/train_high_rank_eigenmodel.py  \
        --wandb-project high-rank-eigenmodel-rank${i} --model-path outputs/toy_models/high_rank.pt \
        --checkpoint-path outputs/eigenmodels/high_rank_eigenmodel_rank${i}.pt   \
        --n-eigenfeatures 6  --n-eigenrank $i   --top-k .1 \
        --n-training-datapoints 10000   --n-eval-datapoints 1000  --sparsity 0.1 \
        --lr 0.01   --batch-size 32   --lr-step-epochs 100 --lr-decay-rate 0.8 \
        --epochs 1000  --log-epochs 100 --checkpoint-epochs 100 --correlation-set-size 4
done
