import os
import random
import numpy as np

def format_elapsed_time(seconds: float) -> str:
    """
    Format elapsed runtime as hh:mm:ss.

    Parameters
    ----------
    seconds : float
        Runtime in seconds.

    Returns
    -------
    str
        Runtime formatted as hours, minutes, and seconds.
    """
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def set_seed(seed: int = 42) -> None:
    """
    Set the seed for system-level and NumPy random generation.

    Parameters
    ----------
    seed : int, optional
        Random seed used for reproducibility. The default is 42.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def combine_fair_pca_split_representation(
    Z0: np.ndarray,
    Z1: np.ndarray,
    A: np.ndarray,
) -> np.ndarray:
    """
    Recombine group-specific Fair PCA representations into the original row order.

    Fair PCA returns separate reduced representations for the unprotected group
    and protected group. This function reconstructs a single representation matrix
    by placing the rows of ``Z0`` where ``A == 0`` and the rows of ``Z1`` where
    ``A == 1``.

    Parameters
    ----------
    Z0 : np.ndarray
        Reduced representation for observations with ``A == 0``.
    Z1 : np.ndarray
        Reduced representation for observations with ``A == 1``.
    A : np.ndarray
        Sensitive-variable vector used to determine the original row positions.

    Returns
    -------
    np.ndarray
        Combined reduced representation in the original row order.

    Raises
    ------
    ValueError
        If the input dimensions are invalid, if ``Z0`` and ``Z1`` do not have
        the same number of columns, or if their row counts do not match the
        group counts in ``A``.
    """
    A = np.asarray(A)

    if A.ndim != 1:
        raise ValueError("A must be a one-dimensional vector.")

    if Z0.ndim != 2 or Z1.ndim != 2:
        raise ValueError("Z0 and Z1 must be two-dimensional matrices.")

    if Z0.shape[1] != Z1.shape[1]:
        raise ValueError("Z0 and Z1 must have the same number of columns.")

    mask_0 = A == 0
    mask_1 = A == 1

    if np.sum(mask_0) != Z0.shape[0]:
        raise ValueError("The number of rows in Z0 does not match the number of A == 0 rows.")

    if np.sum(mask_1) != Z1.shape[0]:
        raise ValueError("The number of rows in Z1 does not match the number of A == 1 rows.")

    Z = np.empty((A.shape[0], Z0.shape[1]), dtype=float)
    Z[mask_0] = Z0
    Z[mask_1] = Z1

    return Z


def create_folds(
    X: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split feature, sensitive-variable, and label data into cross-validation folds.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    A : np.ndarray
        Sensitive-variable vector of shape (n_samples,).
    Y : np.ndarray
        Label vector of shape (n_samples,).
    n_folds : int, optional
        Number of cross-validation folds. The default is 5.
    seed : int, optional
        Random seed used for reproducibility. The default is 42.

    Returns
    -------
    folds : list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        List containing tuples of the form
        (X_train_fold, A_train_fold, Y_train_fold,
         X_val_fold, A_val_fold, Y_val_fold).
    """
    set_seed(seed)

    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    if not isinstance(A, np.ndarray):
        raise TypeError("A must be a NumPy array.")

    if not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a NumPy array.")

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")

    if A.ndim != 1:
        raise ValueError("A must be a one-dimensional array.")

    if Y.ndim != 1:
        raise ValueError("Y must be a one-dimensional array.")

    if X.shape[0] != A.shape[0] or X.shape[0] != Y.shape[0]:
        raise ValueError("X, A, and Y must contain the same number of observations.")

    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")

    if n_folds > X.shape[0]:
        raise ValueError("n_folds cannot exceed the number of observations.")

    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    fold_indices = np.array_split(indices, n_folds)

    folds = []

    for i in range(n_folds):
        val_indices = fold_indices[i]
        train_indices = np.concatenate(
            [fold_indices[j] for j in range(n_folds) if j != i]
        )

        X_train_fold = X[train_indices]
        A_train_fold = A[train_indices]
        Y_train_fold = Y[train_indices]

        X_val_fold = X[val_indices]
        A_val_fold = A[val_indices]
        Y_val_fold = Y[val_indices]

        folds.append((
            X_train_fold,
            A_train_fold,
            Y_train_fold,
            X_val_fold,
            A_val_fold,
            Y_val_fold
        ))

    return folds

def binary_log_loss(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute the binary log loss between true labels and predicted scores.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_samples,), encoded as 0 and 1.
    scores : np.ndarray
        Predicted positive-class probabilities of shape (n_samples,).

    Returns
    -------
    float
        Binary log loss. Lower values indicate better probabilistic predictions.
    """
    if not isinstance(y_true, np.ndarray):
        raise TypeError("y_true must be a NumPy array.")

    if not isinstance(scores, np.ndarray):
        raise TypeError("scores must be a NumPy array.")

    if y_true.ndim != 1:
        raise ValueError("y_true must be a one-dimensional array.")

    if scores.ndim != 1:
        raise ValueError("scores must be a one-dimensional array.")

    if y_true.shape[0] != scores.shape[0]:
        raise ValueError("y_true and scores must have the same length.")

    unique_labels = np.unique(y_true)
    if not np.all(np.isin(unique_labels, [0, 1])):
        raise ValueError("y_true must only contain binary labels 0 and 1.")

    y_true = y_true.astype(np.float64, copy=False)
    scores = scores.astype(np.float64, copy=False)

    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain NaN or infinite values.")

    eps = 1e-7
    scores = np.clip(scores, eps, 1.0 - eps)

    log_loss = -np.mean(
        y_true * np.log(scores)
        + (1.0 - y_true) * np.log1p(-scores)
    )

    if not np.isfinite(log_loss):
        raise ValueError("binary log loss is not finite.")

    return float(log_loss)