import os
import gc
import torch
import argparse
import numpy as np
import sys
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformer_lens.utils import tokenize_and_concatenate
from eigenestimation.eigenmodel.eigenmodel import EigenModel
from tqdm import tqdm
# Function to load model


def compute_circuit_vals(eigenmodel, dataloader, iters, jac_chunk_size=None):
    circuit_vals, X_ordered = [], []

    gc.collect()
    torch.cuda.empty_cache()

    with torch.no_grad():
        for X_batch in tqdm(dataloader, desc="Computing circuit values"): # Show progress bar of for loop
            X_ordered.append(X_batch)
            each_circuit_val = torch.zeros(X_batch.shape[0], eigenmodel.n_features)
            for _ in range(iters):
                grads = eigenmodel.compute_gradients(X_batch.to('cuda' if torch.cuda.is_available() else 'cpu'), chunk_size=jac_chunk_size)
                each_circuit_val += abs(eigenmodel(grads))[:, :eigenmodel.n_features].to(each_circuit_val.device)
                gc.collect()
                torch.cuda.empty_cache()
            circuit_vals.append(each_circuit_val)
            print(torch.cuda.torch.cuda.max_memory_allocated() / 1e9, 'cuda max memory allocated - after append circuit vals')

    return torch.concat(circuit_vals, dim=0) / iters, torch.concat(X_ordered, dim=0)



