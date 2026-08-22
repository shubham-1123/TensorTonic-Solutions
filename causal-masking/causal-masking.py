import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    """
    Return a causally masked copy of the attention scores.
    """
    # Write code here
    scores = np.asarray(scores, dtype=float).copy()
    n = scores.shape[-1]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    scores[..., mask] = mask_value

    return scores
                