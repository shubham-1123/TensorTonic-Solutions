import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    matrix = np.empty((seq_len, d_model), dtype = float)

    for pos in range(seq_len):
        for i in range(d_model//2):
            div = base**(2*i/d_model)
            matrix[pos][2*i] = np.sin(pos/div)
            matrix[pos][2*i+1] = np.cos(pos/div)

             # Handle odd d_model
            
        if d_model % 2 == 1:
            i = d_model // 2
            div = base ** (2 * i / d_model)
            matrix[pos][2 * i] = np.sin(pos / div)


    return matrix
            