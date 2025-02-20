import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import einops
from typing import Tuple
import itertools



def GenerateCorrelatedData(
    num_features: int, 
    num_datapoints: int, 
    sparsity: float, 
    batch_size: int,
    correlation_set_size: int,
    coefs: torch.Tensor,
   # device: str
) -> Tuple[torch.Tensor, torch.Tensor, DataLoader]:
    
    # Initialize feature vectors with random values
    feature_vectors: np.ndarray = np.random.rand(num_datapoints, num_features)
    # Apply sparsity to the feature vectors
    
    
    sparsity_mask = einops.repeat((np.random.rand(num_datapoints, int(num_features/correlation_set_size)) < sparsity), 'sample feature -> sample (feature s)', s=correlation_set_size)
    
    
    feature_vectors = feature_vectors * sparsity_mask

    # Remove feature vectors that are all zeros
    non_zero_indices: np.ndarray = np.any(feature_vectors != 0, axis=1)
    feature_vectors = feature_vectors[non_zero_indices]
    

    # Convert feature vectors to numpy arrays
    data_inputs: np.ndarray = np.array(feature_vectors)
     
    # Create tensors from numpy arrays and move to GPU
    X_tms: torch.Tensor = torch.tensor(data_inputs, dtype=torch.float32)
    
    Y_tms = einops.einsum(X_tms, coefs, 's f, f o -> s o')


    # Create a DataLoader for the dataset
    dataloader_tms: DataLoader = DataLoader(
        TensorDataset(X_tms, Y_tms), 
        batch_size=batch_size, 
        shuffle=True, 
    )

    return X_tms, Y_tms, dataloader_tms

