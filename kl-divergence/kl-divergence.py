import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    # Write code here
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p_safe = np.clip(p, eps, None)
    q_safe = np.clip(q, eps, None)

    return np.sum(p_safe * np.log(p_safe/q_safe))

    