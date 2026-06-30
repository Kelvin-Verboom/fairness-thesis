"""
Reconstruction evaluation utilities for dimensionality reduction methods.

This module provides functions for computing reconstruction error and
reconstruction loss. The reconstruction loss is measured relative to the
optimal rank-k approximation obtained through standard PCA.
"""

import numpy as np

from preprocessing.standard_pca import StandardPCA

def reconstruct_error(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute the reconstruction error as the squared Frobenius norm of A - B.

    Parameters
    ----------
    A : np.ndarray
        Original matrix.
    B : np.ndarray
        Reconstructed matrix.

    Returns
    -------
    float
        Squared Frobenius norm (||A - B||_F)^2.
    """
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise TypeError("A and B must both be NumPy arrays.")

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must both be two-dimensional matrices.")

    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape.")

    return np.linalg.norm(A - B, ord="fro") ** 2

def reconstruct_loss(A: np.ndarray, B: np.ndarray, k: int) -> float:
    """
    Compute the reconstruction loss of B relative to the optimal rank-k
    approximation of A.

    The loss is defined as:

        ||A - B||_F^2 - ||A - A_hat||_F^2

    where A_hat is the optimal rank-k PCA approximation of A.

    Parameters
    ----------
    A : np.ndarray
        Original matrix.
    B : np.ndarray
        Candidate reconstruction matrix with the same shape as A.
    k : int
        Rank of the optimal PCA approximation.

    Returns
    -------
    float
        Reconstruction loss relative to the optimal rank-k approximation.
    """
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise TypeError("A and B must both be NumPy arrays.")

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must both be two-dimensional matrices.")

    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape.")

    if not isinstance(k, int):
        raise TypeError("k must be an integer.")

    if k <= 0 or k > A.shape[1]:
        raise ValueError("k must satisfy 1 <= k <= number of features.")

    standard_pca = StandardPCA(k)
    A_optimal = standard_pca.fit_reconstruct(A)

    return reconstruct_error(A, B) - reconstruct_error(A, A_optimal)