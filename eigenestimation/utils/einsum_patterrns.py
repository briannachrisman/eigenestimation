import einops
import torch

def generate_einsum_pattern(n):
    """
    Generates an einsum pattern string dynamically for `n` rank indices.
    r1 r2 r3 f, r1 d1 f, r2 d2 f, r3 d3 f -> d1 d2 d3 f
    """
    r_indices = [f"r{i+1}" for i in range(n)]  # ['r1', 'r2', 'r3']
    d_indices = [f"d{i+1}" for i in range(n)]  # ['d1', 'd2', 'd3']
    
    # Left-hand side of the equation (inputs)
    input_terms = [f"{' '.join(r_indices)} f"] + [f"{r} {d} f" for r, d in zip(r_indices, d_indices)]
    
    # Right-hand side of the equation (output)
    output_term = f"{' '.join(d_indices)} f"
    
    # Combine into final einsum pattern
    pattern = f"{', '.join(input_terms)} -> {output_term}"
    
    return pattern


def generate_forward_einsum_pattern(n):
    """
    Generates an einsum pattern string dynamically for arbitrary `n` rank indices.
    r1 r2 r3 f, r1 d1 f, r2 d2 f, r3 d3 f, b d1 d2 d3 -> b f
 -> b f

    """
    r_indices = [f"r{i+1}" for i in range(n)]  # ['r1', 'r2', 'r3']
    d_indices = [f"d{i+1}" for i in range(n)]  # ['d1', 'd2', 'd3']
    
    # Left-hand side of the equation (input tensors)
    input_terms = [
        f"{' '.join(r_indices)} f",  # Second term: r1, r2, r3
    ] + [f"{r} {d} f" for r, d in zip(r_indices, d_indices)] + [
        f"b {' '.join(d_indices)}"]  # First term: batch + d1, d2, d3
# Terms for rank-dimension interaction
    
    # Right-hand side of the equation (output tensor)
    output_term = "b f"
    
    # Combine into einsum pattern
    pattern = f"{', '.join(input_terms)} -> {output_term}"
    
    return pattern



def generate_reconstruct_einsum_pattern(n):
    """
    Generates an einsum pattern string dynamically for arbitrary `n` rank indices.
    b f, r1 r2 r3 f, r1 d1 f, r2 d2 f, r3 d3 f -> b d1 d2 d3
    """
    r_indices = [f"r{i+1}" for i in range(n)]  # ['r1', 'r2', 'r3']
    d_indices = [f"d{i+1}" for i in range(n)]  # ['d1', 'd2', 'd3']
    
    # Left-hand side of the equation (input tensors)
    input_terms = [
        "b f",  # First term: batch and feature dimension
        f"{' '.join(r_indices)} f",  # Second term: rank indices (r1, r2, r3, ...)
    ] + [f"{r} {d} f" for r, d in zip(r_indices, d_indices)]
# Terms for rank-dimension interaction
    
    # Right-hand side of the equation (output tensor)
    output_term = f"b {' '.join(d_indices)}"
    
    # Combine into einsum pattern
    pattern = f"{', '.join(input_terms)} -> {output_term}"
    
    return pattern


def generate_add_to_network_einsum_pattern(n):
    """
    Generates an einsum pattern string dynamically for arbitrary `n` rank indices.
    f, r1 r2 r3 f, r1 d1 f, r2 d2 f, r3 d3 f -> d1 d2 d3
    """
    r_indices = [f"r{i+1}" for i in range(n)]  # ['r1', 'r2', 'r3']
    d_indices = [f"d{i+1}" for i in range(n)]  # ['d1', 'd2', 'd3']
    
    # Left-hand side of the equation (input tensors)
    input_terms = [
        "f",  # First term: feature dimension
        f"{' '.join(r_indices)} f",  # Second term: rank indices with 'f' (e.g., 'r1 r2 r3 f')
    ] + [f"{r} {d} f" for r, d in zip(r_indices, d_indices)]  # Terms for rank-dimension interaction
    
    # Right-hand side of the equation (output tensor)
    output_term = f"{' '.join(d_indices)}"
    
    # Combine into einsum pattern
    pattern = f"{', '.join(input_terms)} -> {output_term}"
    
    return pattern


def generate_reconstruct_network_einsum_pattern(n):
    """
    Generates an einsum pattern string dynamically for arbitrary `n` rank indices.
    r1 r2 r3 f, r1 d1 f, r2 d2 f, r3 d3 f -> d1 d2 d3
    """
    r_indices = [f"r{i+1}" for i in range(n)]  # ['r1', 'r2', 'r3']
    d_indices = [f"d{i+1}" for i in range(n)]  # ['d1', 'd2', 'd3']
    
    # Left-hand side of the equation (input tensors)
    input_terms = [
        f"{' '.join(r_indices)} f",  # Second term: rank indices with 'f' (e.g., 'r1 r2 r3 f')
    ] + [f"{r} {d} f" for r, d in zip(r_indices, d_indices)]  # Terms for rank-dimension interaction
    
    # Right-hand side of the equation (output tensor)
    output_term = f"{' '.join(d_indices)}"
    
    # Combine into einsum pattern
    pattern = f"{', '.join(input_terms)} -> {output_term}"
    
    return pattern