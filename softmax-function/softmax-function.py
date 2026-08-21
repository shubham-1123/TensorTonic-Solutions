import numpy as np

def softmax(x: list) -> np.ndarray:
    """Return stable softmax probabilities with the same shape as x."""
    # Write code here
    x = np.asarray(x)
    shifted = x - np.max(x, axis=-1, keepdims = True)
    exp_x = np.exp(shifted)
    return exp_x/np.sum(exp_x, axis=-1, keepdims=True)