import torch
import psutil
from typing import Dict, List

def log_memory_usage(stage: str):
    """Log current memory usage"""
    used = psutil.virtual_memory().used / (1024 ** 2)  # MB
    print(f"Memory at {stage}: {used:.1f}MB")

def pad_parameters_mem_efficient(tensor: torch.Tensor, 
                                 target_shape: tuple, 
                                 method: str = 'mean') -> torch.Tensor:
    """
    Memory-efficient context-based padding for LoRA parameters.
    Pads the tensor to match target_shape using mean or adjacency-based padding.
    """
    padded = torch.zeros(target_shape, device=tensor.device, dtype=tensor.dtype)
    padded[:tensor.shape[0], :tensor.shape[1]] = tensor

    if method == 'mean':
        # Row padding (vertical)
        if target_shape[0] > tensor.shape[0]:
            row_mean = tensor.mean(dim=0, keepdim=True)  # [1, D]
            padded[tensor.shape[0]:] = row_mean.expand(
                target_shape[0] - tensor.shape[0], tensor.shape[1])

        # Column padding (horizontal)
        if target_shape[1] > tensor.shape[1]:
            col_mean = tensor[:, -1:].mean(dim=1, keepdim=True)  # or adjust dim accordingly

            padded[:, tensor.shape[1]:] = col_mean.expand(
            padded.shape[0], target_shape[1] - tensor.shape[1])

    elif method == 'adjacency':
        # Copy adjacent row/column
        if target_shape[0] > tensor.shape[0]:
            padded[tensor.shape[0]:] = tensor[-1:, :]
        if target_shape[1] > tensor.shape[1]:
            padded[:, tensor.shape[1]:] = padded[:, tensor.shape[1]-1:tensor.shape[1]]

    return padded


def safe_load_state_dict(model: torch.nn.Module, state_dict: Dict):
    """
    Load state dict in chunks to avoid memory spikes
    """
    current_state = model.state_dict()
    
    # Update in smaller groups
    keys = list(state_dict.keys())
    chunk_size = max(1, len(keys) // 4)  # Process in 4 chunks
    
    for i in range(0, len(keys), chunk_size):
        chunk_keys = keys[i:i + chunk_size]
        update_dict = {k: state_dict[k] for k in chunk_keys if k in current_state}
        
        # Load this chunk
        current_state.update(update_dict)
        model.load_state_dict(current_state, strict=False)
        
        # Clean up
        del update_dict
        torch.cuda.empty_cache() if torch.cuda.is_available() else None