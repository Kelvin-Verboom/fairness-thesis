"""
Binary multi-layer perceptron classifier implemented with PyTorch.

This module defines a wrapper around a PyTorch binary neural network model. The
wrapper is used for downstream classification experiments and provides methods
for fitting the model and predicting positive-class scores or binary labels.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from classifiers.mlp_utils import BinaryMLP

@dataclass
class MLPClassifier:
    """
    Binary multi-layer perceptron classifier for downstream prediction tasks.
    
    This class wraps a PyTorch neural network for binary classification. It
    provides methods for fitting the model and predicting positive-class scores
    or binary labels.
    
    Parameters
    ----------
    threshold : float, default=0.5
        Decision threshold used to convert predicted positive-class scores into
        binary labels.
    hidden_size : int, default=64
        Number of neurons in each hidden layer.
    epochs : int, default=100
        Number of training epochs.
    batch_size : int, default=32
        Number of observations used in each training batch.
    learning_rate : float, default=0.001
        Learning rate used by the Adam optimizer.
    random_state : Optional[int], default=42
        Random seed used for NumPy and PyTorch operations. If None, no seed is
        set by the wrapper.
    device : Optional[str], default=None
        Device used for model training and prediction. If None, CUDA is used
        when available; otherwise, CPU is used.
    shuffle : bool, default=False
        Whether to shuffle observations in the training dataloader.
    verbose : bool, default=False
        Whether to print training progress during fitting.
    """

    threshold: float = 0.5
    hidden_size: int = 64
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    random_state: Optional[int] = 42
    device: Optional[str] = None
    shuffle: bool = False
    verbose: bool = False
    model: Optional[BinaryMLP] = field(init=False, default=None)
    fitted: bool = field(init=False, default=False)
    input_size: Optional[int] = field(init=False, default=None)
    torch_device: torch.device = field(init=False)

    def __post_init__(self) -> None:
        """Validate settings and initialize the PyTorch device."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")

        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be strictly positive.")

        if self.epochs <= 0:
            raise ValueError("epochs must be strictly positive.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be strictly positive.")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be strictly positive.")

        if self.device is None:
            self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.torch_device = torch.device(self.device)

        if self.random_state is not None:
            self._set_seed(self.random_state)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
        """
        Fit the MLP classifier on the training data.
        
        Parameters
        ----------
        X_train : np.ndarray
            Two-dimensional feature matrix containing the training observations.
        y_train : np.ndarray
            One-dimensional binary label vector containing the training labels.
        
        Returns
        -------
        MLPClassifier
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

        if self.random_state is not None:
            self._set_seed(self.random_state)

        X_train_float = X_train.astype(np.float32, copy=False)
        y_train_float = y_train.astype(np.float32, copy=False)

        train_loader = self._prepare_dataloader(
            X_train_float,
            y_train_float,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

        self.input_size = X_train.shape[1]
        self.model = BinaryMLP(self.input_size, hidden_size=self.hidden_size).to(self.torch_device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.torch_device)
                y_batch = y_batch.to(self.torch_device)
            
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                epoch_loss += loss.item()

            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

        self.fitted = True
        return self

    def predict_scores(self, X_test: np.ndarray) -> np.ndarray:
        """
        Return predicted binary labels for test observations.

        Parameters
        ----------
        X_test : np.ndarray
            Two-dimensional feature matrix containing the test observations.
        threshold : Optional[float], default=None
            Decision threshold applied to the predicted positive-class scores.
            If None, the classifier's default threshold is used.

        Returns
        -------
        np.ndarray
            One-dimensional array containing the predicted binary labels.
        """
        self._check_is_fitted()
        self._validate_features(X_test, name="X_test")

        if X_test.shape[1] != self.input_size:
            raise ValueError("X_test must have the same number of features as the training data.")

        X_tensor = torch.tensor(X_test.astype(np.float32, copy=False), dtype=torch.float32)
        X_tensor = X_tensor.to(self.torch_device)

        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            scores = self.model.predict_prob(X_tensor)

        return scores.cpu().numpy().ravel()

    def predict_labels(self, X_test: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Return binary labels for test data based on a decision threshold.

        Parameters
        ----------
        X_test : np.ndarray
            Test feature matrix of shape (n_samples, n_features).
        threshold : float or None, optional
            Threshold used to convert scores into binary labels. If None, the
            classifier's default threshold is used.

        Returns
        -------
        np.ndarray
            Binary predicted labels of shape (n_samples,), encoded as 0 and 1.
        """
        threshold_value = self.threshold if threshold is None else threshold

        if not 0.0 <= threshold_value <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")

        scores = self.predict_scores(X_test)
        return (scores > threshold_value).astype(int)

    def predict(self, X_test: np.ndarray, threshold: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Return positive-outcome scores and binary labels for test data.

        Parameters
        ----------
        X_test : np.ndarray
            Test feature matrix of shape (n_samples, n_features).
        threshold : float or None, optional
            Threshold used to convert scores into binary labels. If None, the
            classifier's default threshold is used.

        Returns
        -------
        scores : np.ndarray
            Positive-class scores of shape (n_samples,).
        labels : np.ndarray
            Binary predicted labels of shape (n_samples,), encoded as 0 and 1.
        """
        threshold_value = self.threshold if threshold is None else threshold

        if not 0.0 <= threshold_value <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")

        scores = self.predict_scores(X_test)
        labels = (scores > threshold_value).astype(int)

        return scores, labels

    def _check_is_fitted(self) -> None:
        """Check whether the classifier has been fitted."""
        if not self.fitted or self.model is None or self.input_size is None:
            raise ValueError("This MLPClassifier instance is not fitted yet. Call train first.")

    @staticmethod
    def _prepare_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
        """
        Create a PyTorch dataloader from feature and label arrays.

        Parameters
        ----------
        X : np.ndarray
            Two-dimensional feature matrix containing the training observations.
        y : np.ndarray
            One-dimensional binary label vector containing the training labels.
        batch_size : int
            Number of observations used in each batch.
        shuffle : bool
            Whether to shuffle observations before creating batches.

        Returns
        -------
        DataLoader
            PyTorch dataloader containing feature and label tensors.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def _set_seed(seed: int) -> None:
        """Set NumPy and PyTorch seeds."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
