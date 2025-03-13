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
def load_eigenmodel(model_path):
    model_data = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    return model_data['model'], model_data['frac_activated']

# Function to load dataset
def load_and_tokenize_dataset(dataset_name, split, tokenizer, token_length, num_samples):
    dataset = load_dataset(dataset_name, split=split)
    X_train = tokenize_and_concatenate(dataset, tokenizer, max_length=token_length, add_bos_token=False)['tokens']
    return X_train[torch.randperm(len(X_train)), :][:num_samples]  # Shuffle and select subset

# Function to compute circuit values
def compute_circuit_vals(eigenmodel, dataloader, iters, jac_chunk_size=None):
    circuit_vals, X_ordered = [], []

    gc.collect()
    torch.cuda.empty_cache()

    with torch.no_grad():
        for X_batch in tqdm(dataloader, desc="Computing circuit values"): # Show progress bar of for loop
            
            X_ordered.append(X_batch)
            each_circuit_val = torch.zeros(X_batch.shape[0] * X_batch.shape[1], eigenmodel.n_features)
            for _ in range(iters):
                grads = eigenmodel.compute_gradients(X_batch.to('cuda' if torch.cuda.is_available() else 'cpu'), chunk_size=jac_chunk_size)
                each_circuit_val += (eigenmodel(grads))[:, :eigenmodel.n_features].to(each_circuit_val.device)
                gc.collect()
                torch.cuda.empty_cache()
            circuit_vals.append(each_circuit_val.view(X_batch.shape[0], X_batch.shape[1], eigenmodel.n_features))
            print(torch.cuda.torch.cuda.max_memory_allocated() / 1e9, 'cuda max memory allocated - after append circuit vals')

    return torch.concat(circuit_vals, dim=0) / iters, torch.concat(X_ordered, dim=0)

# Function to extract top features
def extract_top_features(circuit_vals, X_ordered, tokenizer, frac_activated, token_length, top_n=5):
    feature_data = {}  # Dictionary to store results

    for i in circuit_vals.mean(dim=[0, 1]).argsort(descending=True):

        abs_vals = abs(circuit_vals[..., i])
        top_indices = abs_vals.flatten().argsort(descending=True)

        top_b, top_t = torch.div(top_indices, token_length, rounding_mode='floor'), top_indices % token_length
        top_values = abs_vals[top_b, top_t]

        feature_info = []
        sample_idxs_so_far = set()

        for j in range(len(top_indices)):
            sample_idx, token_idx = top_b[j].item(), top_t[j].item()
            if sample_idx in sample_idxs_so_far or len(sample_idxs_so_far) >= top_n:
                continue

            sample_idxs_so_far.add(sample_idx)
            tokens = X_ordered[sample_idx, :token_idx+1]

            feature_info.append({
                "sample_id": sample_idx,
                "token_id": token_idx,
                "value": round(top_values[j].item(), 3),
            })

        feature_data[i.item()] = {
            "activation": round(frac_activated[i].item(), 3),
            "top_examples": feature_info
        }

    return feature_data

