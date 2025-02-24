import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import einops
from typing import Tuple
import itertools


def GenerateTMSInputs(num_features: int, num_datapoints: int, sparsity: float):
    """
    Continuously samples feature vectors until the desired number of data points is reached.
    """
    feature_vectors_all = []
    while len(feature_vectors_all) < num_datapoints:
        feature_vectors = np.random.rand(num_datapoints, num_features)
        feature_vectors *= (np.random.rand(num_datapoints, num_features) < sparsity)
        non_zero_indices = np.any(feature_vectors != 0, axis=1)
        feature_vectors = feature_vectors[non_zero_indices]
        feature_vectors_all.append(feature_vectors)
    
    feature_vectors_all = np.vstack(feature_vectors_all)[:num_datapoints]
    X_tms = torch.tensor(feature_vectors_all, dtype=torch.float32).to("cuda")
    return X_tms


def GenerateCorrelatedData(
    num_features: int, 
    num_datapoints: int, 
    sparsity: float, 
    correlation_set_size: int,
   # device: str
) -> torch.Tensor:
    
    
    feature_vectors_all = []
    while len(feature_vectors_all) < num_datapoints:
        # Initialize feature vectors with random values
        feature_vectors: np.ndarray = np.random.rand(num_datapoints, num_features)
        # Apply sparsity to the feature vectors
        sparsity_mask = einops.repeat((np.random.rand(num_datapoints, int(num_features/correlation_set_size)) < sparsity), 'sample feature -> sample (feature s)', s=correlation_set_size)
        feature_vectors = feature_vectors * sparsity_mask
        feature_vectors = feature_vectors * sparsity_mask

        # Remove feature vectors that are all zeros
        non_zero_indices: np.ndarray = np.any(feature_vectors != 0, axis=1)
        feature_vectors = feature_vectors[non_zero_indices]
        feature_vectors_all.append(feature_vectors)
    
    feature_vectors_all = np.vstack(feature_vectors_all)[:num_datapoints]
    # Convert feature vectors to numpy arrays
    data_inputs: np.ndarray = np.array(feature_vectors_all)
     
    # Create tensors from numpy arrays and move to GPU
    X_tms: torch.Tensor = torch.tensor(data_inputs, dtype=torch.float32)

    return X_tms

