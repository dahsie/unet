

import numpy as np
import random
import torch
import os

def set_seed(seed: int = 42) -> None:
    """
    Sets the seed for various random number generators to ensure reproducibility of results.

    Parameters:
    -----------
    seed : int, optional
        The seed value to use for all random number generators. Default is 42.
    
    Notes:
    ------
    - This function sets the seed for the following:
        * NumPy random number generator
        * Python's built-in random module
        * PyTorch (for both CPU and GPU if available)
    - Additionally, it configures the CuDNN backend to ensure deterministic behavior.
    - PYTHONHASHSEED is set to the seed value to ensure consistent hashing.
    """
    np.random.seed(seed)           # Seed the NumPy random number generator
    random.seed(seed)              # Seed the built-in random module
    torch.manual_seed(seed)        # Seed PyTorch's CPU random number generator
    torch.cuda.manual_seed(seed)   # Seed PyTorch's GPU random number generator

    # When using CUDA with CuDNN, these settings ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set the environment variable PYTHONHASHSEED to ensure consistent hashing
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print('> SEEDING DONE')
