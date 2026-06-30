"""
Simplified LAFTR implementation through TensorFlow.

This module defines the LAFTR class used in the thesis experiments. The
class validates input data, applies optional centering and scaling, constructs
the TensorFlow LAFTR graph, trains it through LAFTRTrainer, and exposes methods
for transforming data, reconstructing features, and retrieving the fitted model.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf

# Madras et al. (2018) implement LAFTR in TensorFlow 1.x graph mode, using
# placeholders, variable scopes, and explicit Session objects. This implementation
# follows that style through TensorFlow 2.x's compat.v1 interface. This interface
# is not compatible with eager execution, which is why we disable it. 
tf1 = tf.compat.v1
tf1.disable_eager_execution()

try:
    from preprocessing.laftr.networks import LAFTRModel
    from preprocessing.laftr.trainer import LAFTRTrainer
except ImportError:  # Allows running the files directly from one folder.
    from networks import LAFTRModel
    from trainer import LAFTRTrainer

@dataclass
class LAFTR:
    """
    Simplified TensorFlow implementation of LAFTR for preprocessing.

    This class learns an adversarially fair low-dimensional representation of
    the input data. The model first optionally centers and scales the input
    features, then trains a TensorFlow LAFTR graph consisting of an encoder,
    decoder, classifier, and adversary. After fitting, ``transform`` returns
    the learned representation Z, while ``reconstruct`` returns reconstructed
    features on the original feature scale.

    Parameters
    ----------
    k : int
        Dimension of the learned representation Z.
    center : bool, optional
        Whether to center the input features before training. The default is
        True.
    scale : bool, optional
        Whether to scale the centered input features by their standard
        deviation before training. The default is False.
    class_coeff : float, optional
        Coefficient multiplying the classification loss. The default is 1.0.
    recon_coeff : float, optional
        Coefficient multiplying the reconstruction loss. The default is 1.0.
    fair_coeff : float, optional
        Coefficient multiplying the adversarial fairness loss. The default is
        1.0.
    hidden_dim : int, optional
        Width of the hidden layers used in the encoder, decoder, and adversary.
        The default is 32.
    learning_rate : float, optional
        Learning rate used by the TensorFlow Adam optimizers. The default is
        1e-3.
    batch_size : int, optional
        Number of observations used in each mini-batch. The default is 64.
    n_epochs : int, optional
        Maximum number of training epochs. The default is 100.
    patience : int, optional
        Number of epochs without validation log-loss improvement before early
        stopping. The default is 10.
    aud_steps : int, optional
        Number of adversary updates performed after each main-network update.
        The default is 1.
    random_state : int, optional
        Random seed used for TensorFlow initialization and mini-batch shuffling.
        The default is 42.
    activation : str, optional
        Hidden-layer activation function used in the MLP blocks. The default is
        ``"leakyrelu"``.
    verbose : bool, optional
        Whether to print training progress. The default is False.
    """

    # USER INITIALISATION
    k: int
    center: bool = True
    scale: bool = False
    class_coeff: float = 1.0
    recon_coeff: float = 1.0
    fair_coeff: float = 1.0
    hidden_dim: int = 32
    learning_rate: float = 1e-3
    batch_size: int = 64
    n_epochs: int = 100
    patience: int = 10
    aud_steps: int = 1
    random_state: int = 42
    activation: str = "leakyrelu"
    verbose: bool = False

    # STANDARD INITIALISATION
    mean: np.ndarray = field(init=False)
    std: np.ndarray = field(init=False)
    model: LAFTRModel = field(init=False)
    trainer: LAFTRTrainer = field(init=False)
    training_history: dict = field(init=False)
    graph: tf.Graph = field(init=False)
    fitted: bool = field(default=False, init=False)

    def fit(self, X, A, Y, X_val=None, A_val=None, Y_val=None) -> "LAFTR":
        """
        Fit the simplified TensorFlow LAFTR model.

        Parameters
        ----------
        X : np.ndarray
            Training feature matrix of shape (n_samples, n_features).
        A : np.ndarray
            Training sensitive variable vector encoded as 0 and 1.
        Y : np.ndarray
            Training label vector encoded as 0 and 1.
        X_val : np.ndarray or None, optional
            Validation feature matrix.
        A_val : np.ndarray or None, optional
            Validation sensitive variable vector.
        Y_val : np.ndarray or None, optional
            Validation label vector.

        Returns
        -------
        LAFTR
            The fitted LAFTR instance.
        """
        
        # Validate training data
        X = self._validate_features(X)
        A = self._validate_binary_vector(A, name="A")
        Y = self._validate_binary_vector(Y, name="Y")
        self._validate_k(X.shape[1])

        # Validate validation data
        use_validation = X_val is not None and A_val is not None and Y_val is not None
        if use_validation:
            X_val = self._validate_features(X_val)
            A_val = self._validate_binary_vector(A_val, name="A_val")
            Y_val = self._validate_binary_vector(Y_val, name="Y_val")
            if X_val.shape[1] != X.shape[1]:
                raise ValueError("X_val must have the same number of features as X.")

        # If wanted, center and scale data
        X = self._fit_preprocess(X)
        if use_validation:
            X_val = self._apply_preprocess(X_val)

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf1.set_random_seed(self.random_state)

            self.model = LAFTRModel(
                input_dim=X.shape[1],
                latent_dim=self.k,
                hidden_dim=self.hidden_dim,
                class_coeff=self.class_coeff,
                recon_coeff=self.recon_coeff,
                fair_coeff=self.fair_coeff,
                activation=self.activation,
            )

            self.trainer = LAFTRTrainer(
                model=self.model,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                patience=self.patience,
                aud_steps=self.aud_steps,
                random_state=self.random_state,
                verbose=self.verbose,
            )

            self.training_history = self.trainer.fit(
                X_train=X,
                A_train=A,
                Y_train=Y,
                X_val=X_val if use_validation else None,
                A_val=A_val if use_validation else None,
                Y_val=Y_val if use_validation else None,
            )

        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to the reduced LAFTR representation.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Reduced representation Z of shape (n_samples, k).
        """
        self._check_is_fitted()
        X = self._validate_features(X)
        self._validate_feature_count(X)
        X_processed = self._apply_preprocess(X)

        with self.graph.as_default():
            return self.trainer.transform(X_processed)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from its reduced LAFTR representation.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Reconstructed feature matrix of shape (n_samples, n_features).
        """
        self._check_is_fitted()
        X = self._validate_features(X)
        self._validate_feature_count(X)
        X_processed = self._apply_preprocess(X)

        with self.graph.as_default():
            X_hat_processed = self.trainer.reconstruct(X_processed)

        return self._invert_preprocess(X_hat_processed)

    def fit_transform(self, X: np.ndarray, A: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Fit the model and return the learned representation.
    
        Parameters
        ----------
        X : np.ndarray
            Training feature matrix.
        A : np.ndarray
            Training sensitive-variable values.
        Y : np.ndarray
            Training label values.
    
        Returns
        -------
        np.ndarray
            Learned representation Z.
        """
        self.fit(X, A, Y)
        return self.transform(X)

    def fit_reconstruct(self, X: np.ndarray, A: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Fit the model and return reconstructed features.
    
        Parameters
        ----------
        X : np.ndarray
            Training feature matrix.
        A : np.ndarray
            Training sensitive-variable values.
        Y : np.ndarray
            Training label values.
    
        Returns
        -------
        np.ndarray
            Reconstructed feature matrix.
        """
        self.fit(X, A, Y)
        return self.reconstruct(X)

    def get_k(self) -> int:
        """
        Return the target dimension of the learned representation.
    
        Returns
        -------
        int
            Target dimension k.
        """
        return self.k
    
    def get_model(self) -> LAFTRModel:
        """
        Return the fitted TensorFlow LAFTR graph object.
    
        Returns
        -------
        LAFTRModel
            Fitted TensorFlow graph object.
        """
        self._check_is_fitted()
        return self.model
    
    def get_training_history(self) -> dict:
        """
        Return the training history of the fitted model.
    
        Returns
        -------
        dict
            Dictionary containing the recorded training and validation losses.
        """
        self._check_is_fitted()
        return self.training_history
    
    def close(self) -> None:
        """
        Close the TensorFlow session attached to the fitted model.
        """
        if hasattr(self, "trainer") and self.trainer.sess is not None:
            self.trainer.sess.close()

    def _fit_preprocess(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the preprocessing statistics and return the processed feature matrix.
    
        Parameters
        ----------
        X : np.ndarray
            Feature matrix used to compute the centering and scaling statistics.
    
        Returns
        -------
        np.ndarray
            Optionally centered and optionally scaled feature matrix.
        """
        if self.center:
            self.mean = X.mean(axis=0)
        else:
            self.mean = np.zeros(X.shape[1], dtype=np.float32)

        X_processed = X - self.mean

        if self.scale:
            self.std = X_processed.std(axis=0)
            self.std[self.std == 0] = 1.0
            X_processed = X_processed / self.std
        else:
            self.std = np.ones(X.shape[1], dtype=np.float32)

        return X_processed.astype(np.float32)

    def _apply_preprocess(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the fitted preprocessing transformation to a feature matrix.
    
        Parameters
        ----------
        X : np.ndarray
            Feature matrix to preprocess.
    
        Returns
        -------
        np.ndarray
            Centered and scaled feature matrix.
        """
        return ((X - self.mean) / self.std).astype(np.float32)

    def _invert_preprocess(self, X: np.ndarray) -> np.ndarray:
        """
        Transform preprocessed features back to the original feature scale.
    
        Parameters
        ----------
        X : np.ndarray
            Preprocessed feature matrix.
    
        Returns
        -------
        np.ndarray
            Feature matrix on the original scale.
        """
        return X * self.std + self.mean

    def _validate_k(self, n_features: int) -> None:
        """
        Validate the target dimension of the learned representation.
    
        Parameters
        ----------
        n_features : int
            Number of features in the input data.
    
        Raises
        ------
        TypeError
            If ``k`` is not an integer.
        ValueError
            If ``k`` is not strictly between 0 and the number of features.
        """
        if not isinstance(self.k, int):
            raise TypeError("k must be an integer.")
        if self.k <= 0 or self.k >= n_features:
            raise ValueError("k must satisfy 0 < k < number of features.")
    
    def _validate_feature_count(self, X: np.ndarray) -> None:
        """
        Validate that a feature matrix matches the fitted feature dimension.
    
        Parameters
        ----------
        X : np.ndarray
            Feature matrix to validate.
    
        Raises
        ------
        ValueError
            If the number of features differs from the fitted data.
        """
        if X.shape[1] != self.mean.shape[0]:
            raise ValueError("X must have the same number of features as the fitted data.")
    
    def _check_is_fitted(self) -> None:
        """
        Check whether the LAFTR model has already been fitted.
    
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if not self.fitted:
            raise RuntimeError("The LAFTR model must be fitted before this method is used.")
    
    @staticmethod
    def _validate_features(X: np.ndarray) -> np.ndarray:
        """
        Validate and convert a feature matrix.
    
        Parameters
        ----------
        X : np.ndarray
            Feature matrix to validate.
    
        Returns
        -------
        np.ndarray
            Validated feature matrix with dtype ``float32``.
        """
        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError("X must contain numeric values.")
        return arr.astype(np.float32)
    
    @staticmethod
    def _validate_binary_vector(v: np.ndarray, name: str) -> np.ndarray:
        """
        Validate and convert a binary vector.
    
        Parameters
        ----------
        v : np.ndarray
            Vector to validate.
        name : str
            Name used in error messages.
    
        Returns
        -------
        np.ndarray
            Validated binary column vector with dtype ``float32``.
        """
        arr = np.asarray(v)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.reshape(-1)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array or a column vector.")
        unique_values = np.unique(arr)
        if not np.all(np.isin(unique_values, [0, 1])):
            raise ValueError(f"{name} must be binary encoded as 0 and 1.")
        return arr.astype(np.float32).reshape(-1, 1)