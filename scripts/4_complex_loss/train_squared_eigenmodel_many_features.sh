source ~/.eigenestimation/bin/activate
cd ~/eigenestimation/
NNODES=1
N_PROC=4
for i in {5,10,15,20}; do
    if [ -f "outputs/eigenmodels/squared_features${i}.pt" ]; then
        rm outputs/eigenmodels/squared_features${i}.pt
    fi
    torchrun  --nnodes=$NNODES --nproc-per-node=$N_PROC  scripts/4_complex_loss/train_squared_eigenmodel.py  \
        --wandb-project squared-eigenmodel-features${i} \
        --checkpoint-path outputs/eigenmodels/squared_features${i}.pt   \
        --model-path outputs/toy_models/squared.pt   \
        --n-eigenfeatures $i  --n-eigenrank 3   --top-k .2 \
        --n-training-datapoints 100   --n-eval-datapoints 100   \
        --lr 0.01   --batch-size 32   --sparsity .05 --lr-step-epochs 100 --lr-decay-rate 0.8 \
        --epochs 200  --log-epochs 10 --checkpoint-epochs 100 
done
