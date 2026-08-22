import numpy as np

def batch_norm_forward(x: list, gamma: list, beta: list, eps: float = 1e-5) -> np.ndarray:
    """Return the training-time BatchNorm output."""
    # Write code here

    x = np.asarray(x, dtype = float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.array(beta, dtype=float)

    axis = (0,) + tuple(range(2, x.ndim))
    
    mu = np.mean(x, axis=axis, keepdims=True)
    variance = np.mean((x-mu)**2, axis=axis, keepdims=True)

    shape = (1, -1) + (1,) * (x.ndim - 2)
    gamma = gamma.reshape(shape)
    beta = beta.reshape(shape)
    
    x_hat = (x-mu)/ np.sqrt(variance + eps)
    y = gamma * x_hat + beta
    return y