"""
TensorFlow graph components for the simplified LAFTR preprocessing model.

This module defines the MLP building blocks and LAFTRModel graph used to learn
adversarially fair representations for downstream classification experiments.
The implementation adapts the TensorFlow structure of Madras et al. (2018).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

import tensorflow as tf

# Madras et al. (2018) implement LAFTR in TensorFlow 1.x graph mode, using
# placeholders, variable scopes, and explicit Session objects. This implementation
# follows that style through TensorFlow 2.x's compat.v1 interface. This interface
# is not compatible with eager execution, which is why we disable it. 
tf1 = tf.compat.v1
tf1.disable_eager_execution()

def _activation(x: tf.Tensor, activation: str):
    """
    Apply a selected activation function to a TensorFlow tensor.

    The supported activation names follow the TensorFlow implementation style
    used by Madras et al. (2018). In this implementation, the function is used
    for hidden layers in the LAFTR MLP blocks.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to which the activation function is applied.
    activation : str
        Name of the activation function.

    Returns
    -------
    tf.Tensor
        Tensor after applying the selected activation function.
    """
    
    if activation == "softplus":
        return tf.nn.softplus(x)
    if activation == "sigmoid":
        return tf.nn.sigmoid(x)
    if activation == "relu":
        return tf.nn.relu(x)
    if activation in {"leakyrelu", "leaky_relu"}:
        return tf.nn.leaky_relu(x)
    if activation in {"None", "none", "identity"}:
        return x
    raise ValueError(
        "activation must be one of: 'softplus', 'sigmoid', 'relu', "
        "'leakyrelu', or 'identity'."
    )

class MLP:
    """
    Minimal TensorFlow MLP block adapted from the Madras et al. repository.

    Parameters
    ----------
    name : str
        Variable-scope name of the block.
    shapes : sequence of int
        Layer widths, including input and output dimensions.
    activation : str, optional
        Activation function used after hidden layers.
    """

    def __init__(self, name: str, shapes: Sequence[int], activation: str = "leakyrelu") -> None:
        if len(shapes) < 2:
            raise ValueError("shapes must contain at least input and output dimensions.")
        if any(dim <= 0 for dim in shapes):
            raise ValueError("All layer dimensions must be strictly positive.")

        self.name = name
        self.shapes = list(shapes)
        self.activation = activation
        self.weights = self._make_weights_and_biases()

    def _make_weights_and_biases(self) -> dict[int, dict[str, tf.Tensor]]:
        """
        Create trainable TensorFlow weights and biases for each MLP layer.
    
        The layer-wise dictionary structure follows the MLP implementation used 
        in the Madras et al. repository.
    
        Returns
        -------
        dict[int, dict[str, tf.Tensor]]
            Dictionary indexed by layer number. Each entry contains the weight
            matrix ``"w"`` and bias vector ``"b"`` for that layer.
        """
        weights: dict[int, dict[str, tf.Tensor]] = {}

        with tf1.variable_scope(self.name):
            for i in range(len(self.shapes) - 1):
                weights[i] = {}
                weights[i]["w"] = tf1.get_variable(
                    name=f"w{i}",
                    shape=[self.shapes[i], self.shapes[i + 1]],
                    initializer=tf1.glorot_uniform_initializer(),
                )
                weights[i]["b"] = tf1.get_variable(
                    name=f"b{i}",
                    shape=[self.shapes[i + 1]],
                    initializer=tf.zeros_initializer(),
                )

        return weights

    def forward(self, x: tf.Tensor) -> tf.Tensor:
        """
        Apply the MLP block to an input tensor.
    
        Each layer first applies an affine transformation. The selected activation
        function is applied after hidden layers, while the output layer is returned
        without an activation.
    
        Parameters
        ----------
        x : tf.Tensor
            Input tensor passed to the first MLP layer.
    
        Returns
        -------
        tf.Tensor
            Output tensor produced by the final MLP layer.
        """
        previous = x
        n_layers = len(self.weights)

        for layer in range(n_layers):
            z = tf.add(
                tf.matmul(previous, self.weights[layer]["w"]),
                self.weights[layer]["b"],
            )
            is_output_layer = layer == n_layers - 1
            
            if is_output_layer:
                previous = z
            else:
                previous = _activation(z, self.activation)

        return previous


@dataclass
class LAFTRModel:
    """
    TensorFlow graph for a simplified LAFTR preprocessing model.
    
    The graph follows the Madras et al. split into an encoder-classifier-decoder
    component and an adversary. In the original LAFTR model, the decoder may use
    both the learned representation Z and the sensitive variable A. In this
    simplified thesis implementation, the decoder reconstructs X from Z only, so
    the representation used downstream is obtained as Z = encoder(X).
    
    Parameters
    ----------
    input_dim : int
        Number of input features.
    latent_dim : int
        Dimension of the learned representation Z.
    hidden_dim : int, optional
        Width of the hidden layers used in the encoder, decoder, and adversary. If
        set to 0, no hidden layer is used. The default is 32.
    class_coeff : float, optional
        Coefficient multiplying the classification loss. The default is 1.0.
    recon_coeff : float, optional
        Coefficient multiplying the reconstruction loss. The default is 1.0.
    fair_coeff : float, optional
        Coefficient multiplying the adversarial fairness loss. The default is 1.0.
    activation : str, optional
        Hidden-layer activation function used in the MLP blocks. The default is
        ``"leakyrelu"``.
    """

    # USER INITIALISATION
    input_dim: int
    latent_dim: int
    hidden_dim: int = 32
    class_coeff: float = 1.0
    recon_coeff: float = 1.0
    fair_coeff: float = 1.0
    activation: str = "leakyrelu"
    
    # GRAPH INITIALISATION
    X: tf.Tensor = field(init=False)
    Y: tf.Tensor = field(init=False)
    A: tf.Tensor = field(init=False)
    
    Z: tf.Tensor = field(init=False)
    X_hat: tf.Tensor = field(init=False)
    
    Y_hat_logits: tf.Tensor = field(init=False)
    Y_hat: tf.Tensor = field(init=False)
    
    A_hat_logits: tf.Tensor = field(init=False)
    A_hat: tf.Tensor = field(init=False)
    
    class_loss: tf.Tensor = field(init=False)
    recon_loss: tf.Tensor = field(init=False)
    adv_loss: tf.Tensor = field(init=False)
    loss: tf.Tensor = field(init=False)
    
    main_vars: list[tf.Variable] = field(init=False)
    adv_vars: list[tf.Variable] = field(init=False)
    
    
    def __post_init__(self) -> None:
        """ Build the TensorFlow graph after initialization. """
        self._define_placeholders()
        self._build_graph()

    def _define_placeholders(self) -> None:
        """
        Create TensorFlow placeholders for the model inputs.
        
        The placeholders represent the feature matrix X, binary labels Y, and
        sensitive-variable values A used during graph execution.
        """
        self.X = tf1.placeholder(tf.float32, [None, self.input_dim], name="X")
        self.Y = tf1.placeholder(tf.float32, [None, 1], name="Y")
        self.A = tf1.placeholder(tf.float32, [None, 1], name="A")

    def _hidden_layers(self) -> list[int]:
        """
        Return the hidden-layer specification used by the MLP blocks.
    
        Returns
        -------
        list[int]
            Empty list if ``hidden_dim`` is 0; otherwise a single-element list
            containing the hidden-layer width.
        """
        if self.hidden_dim == 0:
            return []
    
        return [self.hidden_dim]

    def _build_graph(self) -> None:
        """
        Build the TensorFlow graph for the simplified LAFTR model.
        
        The graph follows the main Madras et al. structure: an encoder maps X to Z,
        a classifier predicts Y from Z, a decoder reconstructs X, and an adversary
        predicts A from Z. This implementation keeps the core adversarial objective
        but simplifies the original repository structure and reconstructs X from Z
        only.
        """
        
        hidden = self._hidden_layers()

        with tf1.variable_scope("model/enc_cla"):
            encoder = MLP(
                name="inputs_to_latents",
                shapes=[self.input_dim] + hidden + [self.latent_dim],
                activation=self.activation,
            )
            self.Z = encoder.forward(self.X)

            classifier = MLP(
                name="latents_to_class_logits",
                shapes=[self.latent_dim, 1],
                activation=self.activation,
            )
            self.Y_hat_logits = classifier.forward(self.Z)
            self.Y_hat = tf.nn.sigmoid(self.Y_hat_logits, name="Y_hat")

            decoder = MLP(
                name="latents_to_reconstructed_inputs",
                shapes=[self.latent_dim] + hidden + [self.input_dim],
                activation=self.activation,
            )
            self.X_hat = decoder.forward(self.Z)

        with tf1.variable_scope("model/aud"):
            adversary = MLP(
                name="latents_to_sensitive_logits",
                shapes=[self.latent_dim] + hidden + [1],
                activation=self.activation,
            )
            self.A_hat_logits = adversary.forward(self.Z)
            self.A_hat = tf.nn.sigmoid(self.A_hat_logits, name="A_hat")

        self.class_loss_vec = binary_cross_entropy(self.Y, self.Y_hat)
        self.recon_loss_vec = tf.reduce_mean(tf.square(self.X - self.X_hat), axis=1)
        self.adv_loss_vec = binary_cross_entropy(self.A, self.A_hat)

        self.class_loss = tf.reduce_mean(self.class_loss_vec, name="class_loss")
        self.recon_loss = tf.reduce_mean(self.recon_loss_vec, name="recon_loss")
        self.adv_loss = tf.reduce_mean(self.adv_loss_vec, name="adv_loss")

        self.loss = tf.identity(
            self.class_coeff * self.class_loss
            + self.recon_coeff * self.recon_loss
            - self.fair_coeff * self.adv_loss,
            name="laftr_loss",
        )

        self.main_vars = tf1.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES, scope="model/enc_cla")
        self.adv_vars = tf1.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES, scope="model/aud")

def binary_cross_entropy(target: tf.Tensor, pred: tf.Tensor, eps: float = 1e-8) -> tf.Tensor:
    """
    Compute elementwise binary cross-entropy.

    Parameters
    ----------
    target : tf.Tensor
        Binary target tensor.
    pred : tf.Tensor
        Predicted probabilities.
    eps : float, optional
        Small numerical constant used to avoid taking the logarithm of zero.
        The default is 1e-8.

    Returns
    -------
    tf.Tensor
        One-dimensional tensor containing the binary cross-entropy for each
        observation.
    """
    loss = -(
        target * tf.math.log(pred + eps)
        + (1.0 - target) * tf.math.log(1.0 - pred + eps)
    )
    return tf.reshape(loss, [-1])
