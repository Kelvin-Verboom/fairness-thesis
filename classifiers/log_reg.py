"""
Binary logistic regression classifier implemented with scikit-learn.

This module defines a small wrapper around scikit-learn's logistic regression
model. The wrapper is used for downstream classification experiments and
provides methods for fitting the model, retrieving fitted parameters, and
predicting positive-class probabilities or binary labels.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

@dataclass
class LogisticRegressionClassifier:
    """
    Binary logistic regression classifier for downstream prediction tasks.

    This class wraps scikit-learn's logistic regression implementation and
    provides methods for fitting the model, retrieving fitted parameters, and
    predicting positive-class probabilities or binary labels.

    Parameters
    ----------
    threshold : float, default=0.5
        Decision threshold used to convert predicted probabilities into binary
        labels.
    max_iter : int, default=1000
        Maximum number of iterations used by the logistic regression solver.
    random_state : Optional[int], default=42
        Random seed used by the underlying scikit-learn model.
    solver : str, default="lbfgs"
        Optimisation algorithm used by the logistic regression model.
    C : float, default=1.0
        Inverse regularisation strength. Smaller values imply stronger
        regularisation.
    fit_intercept : bool, default=True
        Whether to include an intercept term in the logistic regression model.
    """
    threshold: float = 0.5
    max_iter: int = 1000
    random_state: Optional[int] = 42
    solver: str = "lbfgs"
    C: float = 1.0
    fit_intercept: bool = True
    model: LogisticRegression = field(init=False)
    
    def __post_init__(self) -> None:
        """Initialize the underlying scikit-learn logistic regression model."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")

        if self.C <= 0:
            raise ValueError("C must be strictly positive.")

        self.model = LogisticRegression(
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
            C=self.C,
            fit_intercept=self.fit_intercept,
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegressionClassifier:
        """
        Fit the logistic regression model on the training data.
    
        Parameters
        ----------
        X_train : np.ndarray
            Two-dimensional feature matrix containing the training observations.
        y_train : np.ndarray
            One-dimensional binary label vector containing the training labels.
    
        Returns
        -------
        LogisticRegressionClassifier
            The fitted classifier instance.
    
        Raises
        ------
        TypeError
            If `X_train` or `y_train` is not a NumPy array, or if `X_train`
            does not contain numeric values.
        ValueError
            If `X_train` is not two-dimensional, if `y_train` is not
            one-dimensional and binary encoded, or if both arrays do not contain
            the same number of observations.
        """
        self._validate_features(X_train, name="X_train")
        self._validate_labels(y_train, name="y_train")

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must contain the same number of observations.")

        self.model.fit(X_train, y_train)
        return self    
    
    def get_coefficients(self) -> np.ndarray:
        """
        Return the fitted logistic regression coefficients.
    
        Returns
        -------
        np.ndarray
            One-dimensional array containing the estimated feature coefficients.
        """
        check_is_fitted(self.model)
        return self.model.coef_.ravel()
    
    def get_intercept(self) -> float:
        """
        Return the fitted logistic regression intercept.
    
        Returns
        -------
        float
            Estimated intercept term of the logistic regression model.
        """
        check_is_fitted(self.model)
        return float(self.model.intercept_[0])
    
    def predict_scores(self, X_test: np.ndarray) -> np.ndarray:
        """
        Return predicted positive-class probabilities for test observations.
    
        Parameters
        ----------
        X_test : np.ndarray
            Two-dimensional feature matrix containing the test observations.
    
        Returns
        -------
        np.ndarray
            One-dimensional array containing the predicted probabilities
            P(Y = 1 | X) for each test observation.
        """
        check_is_fitted(self.model)
        self._validate_features(X_test, name="X_test")

        positive_class_index = int(np.where(self.model.classes_ == 1)[0][0])
        return self.model.predict_proba(X_test)[:, positive_class_index]

    def predict_labels(self, X_test: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Return predicted binary labels for test observations.
    
        Parameters
        ----------
        X_test : np.ndarray
            Two-dimensional feature matrix containing the test observations.
        threshold : Optional[float], default=None
            Decision threshold applied to the predicted positive-class
            probabilities. If None, the classifier's default threshold is used.
    
        Returns
        -------
        np.ndarray
            One-dimensional array containing the predicted binary labels.
        """
        threshold_value = self.threshold if threshold is None else threshold

        if not 0.0 <= threshold_value <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")

        scores = self.predict_scores(X_test)
        return (scores > threshold_value).astype(int)
    
    def predict(self, X_test: np.ndarray, threshold: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Return predicted scores and binary labels for test observations.
    
        Parameters
        ----------
        X_test : np.ndarray
            Two-dimensional feature matrix containing the test observations.
        threshold : Optional[float], default=None
            Decision threshold applied to the predicted positive-class
            probabilities. If None, the classifier's default threshold is used.
    
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple containing the predicted positive-class probabilities and the
            corresponding predicted binary labels.
        """
        threshold_value = self.threshold if threshold is None else threshold
    
        if not 0.0 <= threshold_value <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")
    
        scores = self.predict_scores(X_test)
        labels = (scores > threshold_value).astype(int)
    
        return scores, labels

    @staticmethod
    def _validate_features(X: np.ndarray, name: str) -> None:
        """Validate that a feature matrix is a two-dimensional NumPy array."""
        if not isinstance(X, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array.")

        if X.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional array.")

        if not np.issubdtype(X.dtype, np.number):
            raise TypeError(f"{name} must contain numeric values.")

    @staticmethod
    def _validate_labels(y: np.ndarray, name: str) -> None:
        """Validate that labels are one-dimensional and binary encoded."""
        if not isinstance(y, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array.")

        if y.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array.")

        unique_labels = np.unique(y)
        if not np.array_equal(unique_labels, np.array([0, 1])):
            raise ValueError(f"{name} must contain both binary labels 0 and 1.")    
    