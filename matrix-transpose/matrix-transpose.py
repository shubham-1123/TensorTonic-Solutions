import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    x = np.asarray(A)
    rows, cols = x.shape
    y = np.empty(shape=(x.shape[1], x.shape[0]), dtype = float)

    for i in range(rows):
        for j in range(cols):
            y[j][i] = x[i][j]
    
    return y


    
    
