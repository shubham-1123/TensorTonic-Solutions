import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Vectorized sigmoid function.
    """
    # Write code here

    x = np.asarray(x)
    result = np.empty_like(x, dtype = float)
    positive = x >= 0
    result[positive] = 1 / (1 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x/(1 + exp_x)
    return result







# def sigmoid(x: list | float) -> np.ndarray | float:
#     """
#     Vectorized sigmoid function.
#     """
#     # Write code here
#     x = np.asarray(x)
#     result = np.empty_like(x, dtype = float)
#     positive = x >= 0
#     result[positive] = 1/(1 + np.exp(-x[positive]))

#     exp_x = np.exp(x[~positive])
#     result[~positive] = exp_x / (1+exp_x)

#     return result