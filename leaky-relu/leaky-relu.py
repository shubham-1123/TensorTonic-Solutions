import numpy as np

def leaky_relu(x: list | float, alpha: float = 0.01) -> np.ndarray:
    """
    Apply Leaky ReLU elementwise and return a NumPy array.
    """
    # Write code here
    x = np.asarray(x)
    result = np.empty_like(x, dtype = float)
    positive = x>=0

    result[positive] = x[positive]
    result[~positive] = alpha * x[~positive]
    
    return result