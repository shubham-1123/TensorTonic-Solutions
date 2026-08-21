import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here
    x = np.asarray(x)
    result = np.empty_like(x, dtype = float)
    positive = x > 0
    result[positive] = x[positive]
    result[~positive] = 0
    return result