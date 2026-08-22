import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    # Write code here
    matrix = np.asarray(matrix)


    if matrix.ndim != 2:
        return None
        
    if axis is not None and (axis < 0 or axis >= matrix.ndim):
        return None
    
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    if norm_type == 'l2':
        norm =  np.sqrt(np.sum(matrix**2, axis=axis, keepdims=True))
        norm = np.where(norm == 0, 1, norm)
        matrix = matrix/norm
        return matrix
    elif norm_type == 'l1':
        norm = np.sum(np.abs(matrix), axis=axis, keepdims=True)
        norm = np.where(norm == 0, 1, norm)
        matrix = matrix/norm
        return matrix
    elif norm_type == 'max':
        norm =  np.max(np.abs(matrix), axis=axis, keepdims=True)
        norm = np.where(norm == 0, 1, norm)
        return matrix/norm
        
                        