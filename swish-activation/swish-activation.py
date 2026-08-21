import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    x = np.asarray(x)
    result = np.empty_like(x, dtype=float)
    positive = x >= 0
    result[positive] =  x[positive] / (1 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    result[~positive] = x[~positive] * exp_x/ (1 + exp_x)

    return result