"""
Prediction evaluation utilities for binary classification experiments.

This module provides functions for computing overall accuracy, protected-group
accuracy, and prediction fairness metrics based on demographic disparity,
equal opportunity, and equalized odds.
"""

import numpy as np

def accuracy(Y: np.ndarray, Y_pred: np.ndarray) -> float:
    """
    Compute overall prediction accuracy.

    Parameters
    ----------
    Y : np.ndarray
        One-dimensional array containing the true binary labels.
    Y_pred : np.ndarray
        One-dimensional array containing the predicted binary labels.

    Returns
    -------
    float
        Share of correctly predicted observations.
    """
    return float(np.mean(Y == Y_pred))
    
def accuracy_protected(Y: np.ndarray, Y_pred: np.ndarray, A: np.ndarray) -> float:
    """
    Compute prediction accuracy for the protected group.

    Parameters
    ----------
    Y : np.ndarray
        One-dimensional array containing the true binary labels.
    Y_pred : np.ndarray
        One-dimensional array containing the predicted binary labels.
    A : np.ndarray
        One-dimensional sensitive-variable vector, where 1 indicates membership
        of the protected group.

    Returns
    -------
    float
        Share of correctly predicted observations among protected observations.
    """
    return float(np.mean(Y[A == 1] == Y_pred[A == 1]))

def disparity(Y_pred: np.ndarray, A: np.ndarray) -> float:
    """
    Compute the demographic disparity metric.

    The metric is computed as the overall positive prediction rate minus the
    protected-group positive prediction rate.

    Parameters
    ----------
    Y_pred : np.ndarray
        One-dimensional array containing the predicted binary labels.
    A : np.ndarray
        One-dimensional sensitive-variable vector, where 1 indicates membership
        of the protected group.

    Returns
    -------
    float
        Difference between the overall positive prediction rate and the
        protected-group positive prediction rate.
    """
    return float(np.mean(Y_pred) - np.mean(Y_pred[A == 1]))

def opty(Y: np.ndarray, Y_pred: np.ndarray, A: np.ndarray) -> float:
    """
    Compute the equal opportunity difference metric.

    The metric is computed as the overall true positive rate minus the
    protected-group true positive rate.

    Parameters
    ----------
    Y : np.ndarray
        One-dimensional array containing the true binary labels.
    Y_pred : np.ndarray
        One-dimensional array containing the predicted binary labels.
    A : np.ndarray
        One-dimensional sensitive-variable vector, where 1 indicates membership
        of the protected group.

    Returns
    -------
    float
        Difference between the overall positive prediction rate among truly
        positive observations and the corresponding protected-group rate.
    """
    positive_mask = Y == 1
    protected_positive_mask = (Y == 1) & (A == 1)

    LHS = np.mean(Y_pred[positive_mask])
    RHS = np.mean(Y_pred[protected_positive_mask])

    return float(LHS - RHS)


def odds(Y: np.ndarray, Y_pred: np.ndarray, A: np.ndarray) -> float:
    """
    Compute the equalized odds difference metric.

    The metric is computed as the overall false positive rate minus the
    protected-group false positive rate.

    Parameters
    ----------
    Y : np.ndarray
        One-dimensional array containing the true binary labels.
    Y_pred : np.ndarray
        One-dimensional array containing the predicted binary labels.
    A : np.ndarray
        One-dimensional sensitive-variable vector, where 1 indicates membership
        of the protected group.

    Returns
    -------
    float
        Difference between the overall positive prediction rate among truly
        negative observations and the corresponding protected-group rate.
    """
    negative_mask = Y == 0
    protected_negative_mask = (Y == 0) & (A == 1)

    if np.sum(negative_mask) == 0:
        raise ValueError("Y must contain at least one negative observation.")

    if np.sum(protected_negative_mask) == 0:
        raise ValueError("There must be at least one protected negative observation.")

    LHS = np.mean(Y_pred[negative_mask])
    RHS = np.mean(Y_pred[protected_negative_mask])

    return float(LHS - RHS)