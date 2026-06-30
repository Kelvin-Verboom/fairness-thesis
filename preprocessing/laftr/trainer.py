"""
TensorFlow trainer for the simplified LAFTR preprocessing model.

This class handles the TensorFlow session, alternating adversarial training,
loss evaluation, mini-batch construction, and early stopping. It also contains
the graph-execution helper methods ``transform``, ``reconstruct``, and
``predict_scores``, which are called by the public LAFTR class in ``laftr.py``.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import tensorflow as tf

# Madras et al. (2018) implement LAFTR in TensorFlow 1.x graph mode, using
# placeholders, variable scopes, and explicit Session objects. This implementation
# follows that style through TensorFlow 2.x's compat.v1 interface. This interface
# is not compatible with eager execution, which is why we disable it. 
tf1 = tf.compat.v1
tf1.disable_eager_execution()

@dataclass
class LAFTRTrainer:
    """
    TensorFlow trainer for the simplified LAFTR model.

    Training follows the Madras et al. adversarial pattern: first update the
    encoder-classifier-decoder variables, then update the adversary variables.

    Parameters
    ----------
    model : object
        Fitted TensorFlow graph object containing the LAFTR losses, placeholders,
        and trainable-variable groups.
    learning_rate : float, optional
        Learning rate used by the Adam optimizers. The default is 1e-3.
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
        Random seed used for shuffling mini-batches. The default is 42.
    verbose : bool, optional
        Whether to print training progress. The default is False.
    sess : tf.compat.v1.Session, optional
        Existing TensorFlow session. If not provided, a new session is created.
    """

    # USER INITIALISATION
    model: object
    learning_rate: float = 1e-3
    batch_size: int = 64
    n_epochs: int = 100
    patience: int = 10
    aud_steps: int = 1
    random_state: int = 42
    verbose: bool = False
    sess: Optional[tf1.Session] = None

    # TRAINING INITIALISATION
    rng: np.random.RandomState = field(init=False)
    optimizer_main: tf1.train.AdamOptimizer = field(init=False)
    train_main_op: tf.Operation = field(init=False)
    optimizer_adv: tf1.train.AdamOptimizer = field(init=False)
    train_adv_op: tf.Operation = field(init=False)
    history: dict[str, list[float]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.RandomState(self.random_state)

        # Set up optimizer for encoder, decoder and classifier
        self.optimizer_main = tf1.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_main_op = self.optimizer_main.minimize(
            self.model.loss,
            var_list=self.model.main_vars,
        )
        
        # Set up optimizer for adversary
        self.optimizer_adv = tf1.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_adv_op = self.optimizer_adv.minimize(
            -self.model.loss,
            var_list=self.model.adv_vars,
        )

        # Start or restart TensorFlow session
        self.sess = self.sess or tf1.Session()
        
        # Initialise all TensorFlow graph variables before training.
        self.sess.run(tf1.global_variables_initializer())

        self.history = {
            "train_total_loss": [],
            "train_class_loss": [],
            "train_recon_loss": [],
            "train_adv_loss": [],
            "train_log_loss": [],
            "val_total_loss": [],
            "val_class_loss": [],
            "val_recon_loss": [],
            "val_adv_loss": [],
            "val_log_loss": [],
        }

    def fit(
        self,
        X_train: np.ndarray,
        A_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        A_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
    ) -> dict[str, list[float]]:
        """
        Train the LAFTR model using alternating adversarial updates.
    
        For each mini-batch, the encoder-classifier-decoder network is updated
        first, after which the adversary is updated ``aud_steps`` times. If
        validation data are provided, validation log-loss is used for early stopping
        and the best validation model state is restored after training.
    
        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix.
        A_train : np.ndarray
            Training sensitive-variable values.
        Y_train : np.ndarray
            Training label values.
        X_val : np.ndarray, optional
            Validation feature matrix.
        A_val : np.ndarray, optional
            Validation sensitive-variable values.
        Y_val : np.ndarray, optional
            Validation label values.
    
        Returns
        -------
        dict[str, list[float]]
            Training history containing the recorded training and validation loss
            values across epochs.
        """
        
        use_validation = X_val is not None and A_val is not None and Y_val is not None

        best_val_log_loss = np.inf
        best_epoch = -1
        best_state = None

        # Loop over all epochs
        for epoch in range(self.n_epochs):
            
            # Loop over all batches in epoch
            for X_batch, A_batch, Y_batch in self._batch_iterator(X_train, A_train, Y_train):
                feed = self._feed_dict(X_batch, A_batch, Y_batch)
                
                # Train main network
                self.sess.run(self.train_main_op, feed_dict=feed)
                
                # Train adverary
                for _ in range(self.aud_steps):
                    self.sess.run(self.train_adv_op, feed_dict=feed)

            # Evaluate and append history
            train_metrics = self.evaluate(X_train, A_train, Y_train)
            self._append_metrics("train", train_metrics)

            # Optional: Run validation for early stopping
            if use_validation:
                val_metrics = self.evaluate(X_val, A_val, Y_val)
                self._append_metrics("val", val_metrics)

                current_val_log_loss = val_metrics["log_loss"]
                if current_val_log_loss < best_val_log_loss:
                    best_val_log_loss = current_val_log_loss
                    best_epoch = epoch
                    best_state = self._get_model_state()

                if self.verbose:
                    print(
                        f"Epoch {epoch + 1:03d} | "
                        f"train total={train_metrics['total_loss']:.4f} | "
                        f"val log-loss={current_val_log_loss:.4f}"
                    )

                if epoch - best_epoch >= self.patience:
                    if self.verbose:
                        print(
                            f"Early stopping at epoch {epoch + 1}. "
                            f"Best validation log-loss was {best_val_log_loss:.4f} "
                            f"at epoch {best_epoch + 1}."
                        )
                    break
                
            elif self.verbose:
                print(
                    f"Epoch {epoch + 1:03d} | "
                    f"train total={train_metrics['total_loss']:.4f} | "
                    f"train log-loss={train_metrics['log_loss']:.4f}"
                )

        # Update best model state
        if best_state is not None:
            self._set_model_state(best_state)

        return self.history

    def evaluate(self, X: np.ndarray, A: np.ndarray, Y: np.ndarray) -> dict[str, float]:
        """
        Evaluate the LAFTR loss components on a full dataset.
    
        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        A : np.ndarray
            Sensitive-variable values.
        Y : np.ndarray
            Label values.
    
        Returns
        -------
        dict[str, float]
            Dictionary containing the total loss, classification loss,
            reconstruction loss, adversarial loss, and log-loss.
        """
        feed = self._feed_dict(X, A, Y)

        loss_tensors = [
            self.model.loss,
            self.model.class_loss,
            self.model.recon_loss,
            self.model.adv_loss,
        ]
        
        total_loss, class_loss, recon_loss, adv_loss = self.sess.run(
            loss_tensors,
            feed_dict=feed,
        )

        return {
            "total_loss": float(total_loss),
            "class_loss": float(class_loss),
            "recon_loss": float(recon_loss),
            "adv_loss": float(adv_loss),
            "log_loss": float(class_loss),
        }
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Return the learned representation for a feature matrix.
    
        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
    
        Returns
        -------
        np.ndarray
            Learned representation Z.
        """
        n = X.shape[0]
        feed = self._feed_dict(X, np.zeros((n, 1)), np.zeros((n, 1)))
        return self.sess.run(self.model.Z, feed_dict=feed)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Return reconstructed features for a feature matrix.
    
        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
    
        Returns
        -------
        np.ndarray
            Reconstructed feature matrix X_hat.
        """
        n = X.shape[0]
        feed = self._feed_dict(X, np.zeros((n, 1)), np.zeros((n, 1)))
        return self.sess.run(self.model.X_hat, feed_dict=feed)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Return internal classifier scores for a feature matrix.
    
        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
    
        Returns
        -------
        np.ndarray
            Predicted probabilities P(Y=1 | Z).
        """
        n = X.shape[0]
        feed = self._feed_dict(X, np.zeros((n, 1)), np.zeros((n, 1)))
        return self.sess.run(self.model.Y_hat, feed_dict=feed).reshape(-1)

    def _batch_iterator(self, X: np.ndarray, A: np.ndarray, Y: np.ndarray):
        """
        Yield shuffled mini-batches of features, sensitive values, and labels.
    
        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        A : np.ndarray
            Sensitive-variable vector or matrix.
        Y : np.ndarray
            Label vector or matrix.
    
        Yields
        ------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Mini-batch containing features, sensitive values, and labels.
        """
        n = X.shape[0]
        indices = self.rng.permutation(n)
    
        for start in range(0, n, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            yield X[batch_idx], A[batch_idx], Y[batch_idx]

    def _feed_dict(self, X: np.ndarray, A: np.ndarray, Y: np.ndarray) -> dict:
        """
        Create the TensorFlow feed dictionary for graph execution.
    
        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        A : np.ndarray
            Sensitive-variable values.
        Y : np.ndarray
            Label values.
    
        Returns
        -------
        dict
            TensorFlow feed dictionary mapping placeholders to NumPy arrays.
        """
        return {
            self.model.X: np.asarray(X, dtype=np.float32),
            self.model.A: np.asarray(A, dtype=np.float32),
            self.model.Y: np.asarray(Y, dtype=np.float32),
        }

    def _append_metrics(self, prefix: str, metrics: dict[str, float]) -> None:
        """
        Append evaluation metrics to the training history.
    
        Parameters
        ----------
        prefix : str
            Prefix indicating whether the metrics belong to the training or
            validation set.
        metrics : dict[str, float]
            Dictionary containing the loss values returned by ``evaluate``.
        """
        self.history[f"{prefix}_total_loss"].append(metrics["total_loss"])
        self.history[f"{prefix}_class_loss"].append(metrics["class_loss"])
        self.history[f"{prefix}_recon_loss"].append(metrics["recon_loss"])
        self.history[f"{prefix}_adv_loss"].append(metrics["adv_loss"])
        self.history[f"{prefix}_log_loss"].append(metrics["log_loss"])

    def _get_model_state(self) -> list[np.ndarray]:
        """
        Return the current values of all trainable TensorFlow variables.
    
        Returns
        -------
        list[np.ndarray]
            List containing the current numerical values of the trainable variables.
        """
        variables = tf1.trainable_variables()
        return self.sess.run(variables)
    
    def _set_model_state(self, state: list[np.ndarray]) -> None:
        """
        Restore the values of all trainable TensorFlow variables.
    
        Parameters
        ----------
        state : list[np.ndarray]
            List of variable values.
        """
        variables = tf1.trainable_variables()
        assign_ops = [var.assign(value) for var, value in zip(variables, state)]
        self.sess.run(assign_ops)
