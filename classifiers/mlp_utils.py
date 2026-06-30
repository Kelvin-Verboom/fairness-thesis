"""
Binary multi-layer perceptron utilities implemented with PyTorch.

This module defines a small neural network used for binary classification
experiments. The model returns positive-class logits and provides helper
methods for converting these logits into probabilities or binary labels.
"""

from __future__ import annotations

import torch
import torch.nn as nn

class BinaryMLP(nn.Module):
    """
    Two-hidden-layer neural network for binary classification.

    This class defines a feed-forward multi-layer perceptron with two hidden
    layers, ReLU activations, and a single output node. The output is returned
    as a logit for the positive class.

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden_size : int, default=64
        Number of neurons in each hidden layer.
    """

    def __init__(self, input_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return positive-class logits for input observations.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (n_samples, input_size).

        Returns
        -------
        torch.Tensor
            Output logits of shape (n_samples, 1).
        """
        return self.model(x)

    def predict_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return predicted positive-class probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (n_samples, input_size).

        Returns
        -------
        torch.Tensor
            Positive-class probabilities of shape (n_samples, 1), obtained by
            applying the sigmoid function to the output logits.
        """
        return torch.sigmoid(self.forward(x))

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Return predicted binary labels for input observations.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (n_samples, input_size).
        threshold : float, default=0.5
            Decision threshold used to convert predicted probabilities into
            binary labels.

        Returns
        -------
        torch.Tensor
            Binary predicted labels of shape (n_samples, 1), encoded as 0.0
            and 1.0.
        """
        return (self.predict_prob(x) > threshold).float()
