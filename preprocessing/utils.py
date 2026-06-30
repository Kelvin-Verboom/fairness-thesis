"""Utility functions for Fair PCA algorithm."""

import numpy as np

def frobenius_squared(X: np.ndarray) -> float:
    """
    Compute the squared Frobenius norm of a matrix.

    Parameters
    ----------
    X : np.ndarray
        Input matrix.

    Returns
    -------
    float
        Squared Frobenius norm.
    """
    return float(np.linalg.norm(X, ord="fro") ** 2)