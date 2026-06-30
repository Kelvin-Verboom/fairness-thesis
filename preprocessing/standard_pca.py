"""
Standard PCA implementation using eigenvalue decomposition.

This module defines a PCA class that centers numeric input data, computes
principal components from the scatter matrix, and provides methods for
dimensionality reduction and rank-k reconstruction.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class StandardPCA:
    """
    Standard PCA preprocessing model.
    
    This class fits a rank-k PCA projection using eigenvalue decomposition,
    and provides methods for transforming data to the reduced representation
    and reconstructing data from that representation.
    """
    # USER INITIALISATION
    k: int
    center: bool = True

    # STANDARD INITIALISATION
    W: np.ndarray = field(init=False)
    eigenvalues: np.ndarray = field(init=False)
    mean: np.ndarray = field(init=False)
    fitted: bool = field(default=False, init=False)    

    def fit(self, X: np.ndarray) -> StandardPCA:
        """
        Fit PCA on a data matrix using eigenvalue decomposition.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_obs, n_features).

        Returns
        -------
        StandardPCA
            The fitted PCA instance.
        """
        X = self._validate_features(X)        
        
        n_obs, n_features = X.shape
        
        if not isinstance(self.k, int):
            raise TypeError("k must be an integer.")

        if self.k <= 0 or self.k > n_features:
            raise ValueError("k must satisfy 1 <= k <= number of features.")

        if self.center:
            self.mean = X.mean(axis=0)
            X_centered = X - self.mean
        else:
            self.mean = np.zeros(n_features)
            X_centered = X

        S = X_centered.T @ X_centered
        eigenvalues, eigenvectors = np.linalg.eigh(S)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]        
        
        self.eigenvalues = eigenvalues[:self.k]
        self.W = eigenvectors[:, :self.k]
        self.fitted = True
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to the reduced PCA representation.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_obs, n_features).

        Returns
        -------
        np.ndarray
            Reduced representation of shape (n_obs, k).
        """
        self._check_is_fitted()
        X = self._validate_features(X)

        if X.shape[1] != self.W.shape[0]:
            raise ValueError("X must have the same number of features as the fitted data.")

        X_centered = X - self.mean
        return X_centered @ self.W   
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from its reduced PCA representation.
    
        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_obs, n_features).
    
        Returns
        -------
        np.ndarray
            Rank-k PCA reconstruction of X.
        """
        self._check_is_fitted()
    
        Z = self.transform(X)
        X_hat_centered = Z @ self.W.T
    
        if self.center:
            return X_hat_centered + self.mean
    
        return X_hat_centered
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and return the reduced representation.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_obs, n_features).

        Returns
        -------
        np.ndarray
            Reduced representation of shape (n_obs, k).
        """
        self.fit(X)
        return self.transform(X)

    def fit_reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and return the rank-k reconstruction.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_obs, n_features).

        Returns
        -------
        np.ndarray
            Rank-k reconstruction of X.
        """
        self.fit(X)
        return self.reconstruct(X)

    def get_loadings(self) -> np.ndarray:
        """
        Return the fitted PCA loading matrix.

        Returns
        -------
        np.ndarray
            Loading matrix of shape (n_features, k).
        """
        self._check_is_fitted()
        return self.W    
    
    @staticmethod
    def _validate_features(X: np.ndarray) -> np.ndarray:
        """
        Validate and convert a feature matrix.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Feature matrix converted to float.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy ndarray.")

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional matrix.")

        if not np.issubdtype(X.dtype, np.number):
            raise TypeError("X must contain numeric values.")

        return X.astype(float)

    def _check_is_fitted(self) -> None:
        """Check whether the PCA instance has been fitted."""
        if not self.fitted:
            raise RuntimeError("The PCA model must be fitted before this method is used.")