import numpy as np
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    x = np.asarray(x)
    result = np.empty_like(x, dtype=float)
    positive = x>0
    result[positive] = x[positive]
    result[~positive] = alpha * (np.exp(x[~positive]) - 1)
    
    return result.tolist()