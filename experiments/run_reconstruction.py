"""
Run reconstruction experiments for Standard PCA, Fair PCA, and LAFTR.

This module evaluates group-specific reconstruction errors and reconstruction
losses across target dimensions for the COMPAS and Adult datasets. It includes
generic reconstruction runners, method-specific wrappers, result-export helpers,
and command-line utilities for storing reconstruction results in CSV format.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import time
from typing import Iterable, Optional, Callable, Any

import numpy as np
import pandas as pd

# Surpress info and warning messages from TensorFlow.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from preprocessing.standard_pca import StandardPCA
from preprocessing.fair_pca import FairPCA
from preprocessing.laftr.laftr import LAFTR

from evaluation.reconstruction import reconstruct_error, reconstruct_loss
from data.load_data import wrap_COMPAS_full, wrap_Adult_full
from data.load_data import wrap_COMPAS_equal_weights, wrap_Adult_equal_weights

from experiments.utils import format_elapsed_time

# =============================================================================
# GENERIC RECONSTRUCTION RUNNER
# =============================================================================
def run_reconstruction_method(
    method_name: str,
    X: np.ndarray,
    A: np.ndarray,
    k_list: Optional[Iterable[int]],
    reconstruct_for_k: Callable[[int], tuple[np.ndarray, np.ndarray, Any]],
) -> tuple[list[int], list[float], list[float], list[float], list[float], list[Any]]:
    """
    Run reconstruction for one preprocessing method across target dimensions.

    Parameters
    ----------
    method_name : str
        Name of the preprocessing method.
    X : np.ndarray
        Input feature matrix.
    A : np.ndarray
        Sensitive-variable vector, encoded as 0 and 1.
    k_list : Iterable[int] or None
        Target dimensions to evaluate. If None, all dimensions from 1 to n - 1
        are used.
    reconstruct_for_k : Callable[[int], tuple[np.ndarray, np.ndarray, Any]]
        Function that fits the method for one value of k and returns the
        reconstructed matrices for A = 0 and A = 1, plus optional extra output.

    Returns
    -------
    tuple
        Evaluated k values, average errors for A = 0 and A = 1, average losses
        for A = 0 and A = 1, and method-specific extra outputs.
    """
    
    _, n = X.shape

    X_0 = X[A == 0]
    X_1 = X[A == 1]

    if k_list is None:
        k_list = range(1, n)

    k_list = list(k_list)

    avg_errors_0 = []
    avg_errors_1 = []
    avg_losses_0 = []
    avg_losses_1 = []
    extra_outputs = []
    
    method_start_time = time.perf_counter()

    # Run reconstruction for each value of k:
    for k in k_list:
        k_start_time = time.perf_counter()

        # Fit and reconstruct
        X_hat_0, X_hat_1, extra_output = reconstruct_for_k(k)

        # Compute average errors and losses
        avg_error_0, avg_error_1, avg_loss_0, avg_loss_1 = (
            group_evaluation_metrics(
                X_0=X_0,
                X_1=X_1,
                X_hat_0=X_hat_0,
                X_hat_1=X_hat_1,
                k=k,
            )
        )
        
        # Store evaluation metrics
        avg_errors_0.append(avg_error_0)
        avg_errors_1.append(avg_error_1)
        avg_losses_0.append(avg_loss_0)
        avg_losses_1.append(avg_loss_1)
        extra_outputs.append(extra_output)
        
        # Print elapsed time
        k_elapsed_time = time.perf_counter() - k_start_time
        print_k_finished(method_name, k, k_elapsed_time)
    
    # Print elpased time for preprocessing method
    method_elapsed_time = time.perf_counter() - method_start_time
    formatted_method_runtime = format_elapsed_time(method_elapsed_time)
    print(f"{method_name} - total runtime: {formatted_method_runtime}")
    
    return k_list, avg_errors_0, avg_errors_1, avg_losses_0, avg_losses_1, extra_outputs

# =============================================================================
# RECONSTRUCTION RUNNERS FOR RESEARCH METHODS
# =============================================================================
def run_standard_PCA(
    X: np.ndarray,
    A: np.ndarray,
    k_list: Optional[Iterable[int]] = None,
) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    """
    Run Standard PCA reconstruction for a sequence of target dimensions.

    For each value of k, this function fits Standard PCA on the full feature
    matrix, reconstructs the original data, and computes group-specific average
    reconstruction errors and losses.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix.
    A : np.ndarray
        Sensitive-variable vector, where 0 and 1 indicate the two groups.
    k_list : Iterable[int] or None, optional
        Target dimensions to evaluate. If None, all dimensions from 1 to n - 1
        are used.

    Returns
    -------
    tuple[list[int], list[float], list[float], list[float], list[float]]
        Evaluated k values, average reconstruction errors for A = 0 and A = 1,
        and average reconstruction losses for A = 0 and A = 1.
    """

    def reconstruct_for_k(k: int) -> tuple[np.ndarray, np.ndarray, None]:
        # Fit Standard PCA for one target dimension and return reconstructions by group.
        method = StandardPCA(k)
        X_hat = method.fit_reconstruct(X)

        return X_hat[A == 0], X_hat[A == 1], None

    k_values, error_0, error_1, loss_0, loss_1, _ = run_reconstruction_method(
        method_name="Standard PCA",
        X=X,
        A=A,
        k_list=k_list,
        reconstruct_for_k=reconstruct_for_k,
    )

    return k_values, error_0, error_1, loss_0, loss_1

def run_fair_PCA(
    X: np.ndarray,
    A: np.ndarray,
    k_list: Optional[Iterable[int]] = None,
) -> tuple[list[int], list[float], list[float], list[float], list[float], list[int]]:
    """
    Run Fair PCA reconstruction for a sequence of target dimensions.

    For each value of k, this function fits Fair PCA on the full feature matrix
    and sensitive-variable vector, reconstructs the data separately for both
    groups, and computes group-specific average reconstruction errors and losses.
    The fitted Fair PCA rank is also stored for each k.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix.
    A : np.ndarray
        Sensitive-variable vector, where 0 and 1 indicate the two groups.
    k_list : Iterable[int] or None, optional
        Target dimensions to evaluate. If None, all dimensions from 1 to n - 1
        are used.

    Returns
    -------
    tuple[list[int], list[float], list[float], list[float], list[float], list[int]]
        Evaluated k values, average reconstruction errors for A = 0 and A = 1,
        average reconstruction losses for A = 0 and A = 1, and the fitted Fair
        PCA ranks.
    """

    def reconstruct_for_k(k: int) -> tuple[np.ndarray, np.ndarray, int]:
        # Fit Fair PCA for one target dimension and return group reconstructions and rank.
        method = FairPCA(k)
        X_hat_0, X_hat_1 = method.fit_reconstruct_split(X, A)

        return X_hat_0, X_hat_1, method.get_rank_fair()

    k_values, error_0, error_1, loss_0, loss_1, rank_values = run_reconstruction_method(
        method_name="Fair PCA",
        X=X,
        A=A,
        k_list=k_list,
        reconstruct_for_k=reconstruct_for_k,
    )

    return k_values, error_0, error_1, loss_0, loss_1, rank_values

def run_LAFTR(
    X: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    k_list: Optional[Iterable[int]] = None,
    n_epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    class_coeff: float = 1.0,
    recon_coeff: float = 1.0,
    fair_coeff: float = 1.0,
    center: bool = True,
    scale: bool = False,
    hidden_dim: int = 32,
    patience: int = 10,
    aud_steps: int = 1,
    random_state: int = 42,
    activation: str = "leakyrelu",
    verbose: bool = False,
) -> tuple[list[int], list[float], list[float], list[float], list[float], dict]:
    """
    Run LAFTR reconstruction for a sequence of target dimensions.

    For each value of k, this function trains a LAFTR model, reconstructs the
    input data with the fitted decoder, and computes group-specific average
    reconstruction errors and losses. The training history is stored for each
    target dimension.

    The LAFTR-specific configuration parameters are passed directly to the
    ``LAFTR`` class; see the class docstring for their interpretation.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix.
    A : np.ndarray
        Sensitive-variable vector, where 0 and 1 indicate the two groups.
    Y : np.ndarray
        Binary label vector used during LAFTR training.
    k_list : Iterable[int] or None, optional
        Target dimensions to evaluate. If None, all dimensions from 1 to n - 1
        are used.
    n_epochs, batch_size, learning_rate, class_coeff, recon_coeff, fair_coeff,
    center, scale, hidden_dim, patience, aud_steps, random_state, activation,
    verbose
        LAFTR configuration parameters passed directly to the ``LAFTR`` class.

    Returns
    -------
    tuple[list[int], list[float], list[float], list[float], list[float], dict]
        Evaluated k values, average reconstruction errors for A = 0 and A = 1,
        average reconstruction losses for A = 0 and A = 1, and a dictionary of
        LAFTR training histories indexed by k.
    """

    def reconstruct_for_k(k: int) -> tuple[np.ndarray, np.ndarray, tuple[int, dict]]:
        # Fit LAFTR for one target dimension and return group reconstructions and training history.
        method = LAFTR(
            k=k,
            center=center,
            scale=scale,
            class_coeff=class_coeff,
            recon_coeff=recon_coeff,
            fair_coeff=fair_coeff,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_epochs=n_epochs,
            patience=patience,
            aud_steps=aud_steps,
            random_state=random_state,
            activation=activation,
            verbose=verbose,
        )

        method.fit(X, A, Y)
        X_hat = method.reconstruct(X)
        training_history = method.get_training_history()
        method.close()

        return X_hat[A == 0], X_hat[A == 1], (k, training_history)

    k_values, error_0, error_1, loss_0, loss_1, history_items = run_reconstruction_method(
        method_name="LAFTR",
        X=X,
        A=A,
        k_list=k_list,
        reconstruct_for_k=reconstruct_for_k,
    )

    training_histories = dict(history_items)

    return k_values, error_0, error_1, loss_0, loss_1, training_histories

# =============================================================================
# EXPERIMENT HELPER
# =============================================================================
def group_evaluation_metrics(
    X_0: np.ndarray,
    X_1: np.ndarray,
    X_hat_0: np.ndarray,
    X_hat_1: np.ndarray,
    k: int,
) -> tuple[float, float, float, float]:
    """
    Compute average reconstruction errors and losses for both sensitive groups.

    Parameters
    ----------
    X_0 : np.ndarray
        Original feature matrix for the unprotected group.
    X_1 : np.ndarray
        Original feature matrix for the protected group.
    X_hat_0 : np.ndarray
        Reconstructed feature matrix for the unprotected group.
    X_hat_1 : np.ndarray
        Reconstructed feature matrix for the protected group.
    k : int
        Target reconstruction dimension.

    Returns
    -------
    tuple[float, float, float, float]
        Average reconstruction error for A = 0, average reconstruction error
        for A = 1, average reconstruction loss for A = 0, and average
        reconstruction loss for A = 1.
    """
    m0 = X_0.shape[0]
    m1 = X_1.shape[0]

    avg_error_0 = reconstruct_error(X_0, X_hat_0) / m0
    avg_error_1 = reconstruct_error(X_1, X_hat_1) / m1

    avg_loss_0 = reconstruct_loss(X_0, X_hat_0, k) / m0
    avg_loss_1 = reconstruct_loss(X_1, X_hat_1, k) / m1

    return avg_error_0, avg_error_1, avg_loss_0, avg_loss_1

# =============================================================================
# LOGGING HELPER
# =============================================================================
def print_k_finished(method_name: str, k: int, runtime: float) -> None:
    """
    Print progress after one reconstruction run has finished.

    Parameters
    ----------
    method_name : str
        Name of the preprocessing method.
    k : int
        Target dimension used in the reconstruction run.
    runtime : float
        Runtime of the reconstruction run in seconds.
    """
    formatted_runtime = format_elapsed_time(runtime)
    print(f"{method_name} | k={k} - finished in {formatted_runtime}")

# =============================================================================
# RESULT EXPORT HELPERS
# =============================================================================
def initialize_reconstruction_results(results_path: Path, k_values: list[int]) -> pd.DataFrame:
    """
    Load an existing reconstruction results file or initialize a new one.

    The reconstruction results are stored in wide format, with one row per
    target dimension k and separate columns for each method-specific metric. If
    the results file already exists, this function checks whether its k values
    match the current experiment. If no file exists, a new results dataframe is
    initialized.

    Parameters
    ----------
    results_path : Path
        Path to the reconstruction results CSV file.
    k_values : list[int]
        Target dimensions used in the current reconstruction experiment.

    Returns
    -------
    pd.DataFrame
        Existing or newly initialized reconstruction results dataframe.

    Raises
    ------
    ValueError
        If an existing results file does not contain a k column, contains
        duplicate k values, or uses different k values than the current
        experiment.
    """
    k_values = [int(k) for k in k_values]

    if results_path.exists():
        results_df = pd.read_csv(results_path)

        if "k" not in results_df.columns:
            raise ValueError(
                "Existing reconstruction results file does not contain a 'k' column. "
                "Delete the old file or use a different output filename."
            )

        if results_df["k"].duplicated().any():
            raise ValueError(
                "Existing reconstruction results file contains duplicate k values. "
                "Delete the old file or use a different output filename."
            )

        existing_k_values = sorted(results_df["k"].astype(int).tolist())
        current_k_values = sorted(k_values)

        if existing_k_values != current_k_values:
            raise ValueError(
                "Existing reconstruction results file uses different k values. "
                "Delete the old file or use a different output filename."
            )

        results_df["k"] = results_df["k"].astype(int)
        results_df = results_df.sort_values("k").reset_index(drop=True)
        return results_df

    return pd.DataFrame({"k": np.asarray(k_values, dtype=int)})

def store_reconstruction_results(
    results_df: pd.DataFrame,
    method_name: str,
    k_list: list[int],
    avg_errors_0: list[float],
    avg_errors_1: list[float],
    avg_losses_0: list[float],
    avg_losses_1: list[float],
    rank_values: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Store reconstruction results for one preprocessing method.

    The method-specific reconstruction metrics are added to the existing
    results dataframe in wide format. Existing columns for the same method are
    overwritten, while results for other methods are preserved.

    Parameters
    ----------
    results_df : pd.DataFrame
        Reconstruction results dataframe containing one row per target
        dimension k.
    method_name : str
        Name of the preprocessing method used in the output column names.
    k_list : list[int]
        Target dimensions evaluated for the preprocessing method.
    avg_errors_0 : list[float]
        Average reconstruction errors for the unprotected group.
    avg_errors_1 : list[float]
        Average reconstruction errors for the protected group.
    avg_losses_0 : list[float]
        Average reconstruction losses for the unprotected group.
    avg_losses_1 : list[float]
        Average reconstruction losses for the protected group.
    rank_values : list[int] or None, optional
        Fitted ranks to store for the preprocessing method. The default is None.

    Returns
    -------
    pd.DataFrame
        Updated reconstruction results dataframe.

    Raises
    ------
    ValueError
        If the method results contain duplicate k values or if their k values do
        not match those in the existing results dataframe.
    """
    
    # Create results dataframe for this method.
    method_df = pd.DataFrame(
        {
            "k": np.asarray(k_list, dtype=int),
            f"avg_error_A0_{method_name}": avg_errors_0,
            f"avg_error_A1_{method_name}": avg_errors_1,
            f"avg_loss_A0_{method_name}": avg_losses_0,
            f"avg_loss_A1_{method_name}": avg_losses_1,
        }
    )

    # Add method-specific ranks if available.
    if rank_values is not None:
        method_df[f"rank_{method_name}"] = rank_values

    if method_df["k"].duplicated().any():
        raise ValueError(f"{method_name} produced duplicate k values.")

    # Check that the method results match the initialized k values.
    if sorted(method_df["k"].tolist()) != sorted(results_df["k"].astype(int).tolist()):
        raise ValueError(
            f"{method_name} produced k values that do not match the results dataframe."
        )

    # Align both dataframes by target dimension.
    results_df = results_df.set_index("k")
    method_df = method_df.set_index("k")
    
    # Insert or overwrite the method-specific result columns.
    for column in method_df.columns:
        results_df.loc[method_df.index, column] = method_df[column]

    # Restore k as a column and return results sorted by target dimension.
    results_df = results_df.reset_index()
    results_df = results_df.sort_values("k").reset_index(drop=True)

    return results_df

# =============================================================================
# EXPERIMENT WRAPPER
# =============================================================================
def run_reconstruction_experiment(
    dataset_name: str,
    X: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    results_dir: Path,
    k_values: list[int],
    experiment: str,
    run_standard_pca: bool = True,
    run_fair_pca: bool = True,
    run_laftr: bool = True,
) -> None:
    """
    Run reconstruction experiments for one dataset and save the results to CSV.

    Existing method results are preserved. Only the columns of the methods
    selected in the current run are updated.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    X : np.ndarray
        Input feature matrix.
    A : np.ndarray
        Sensitive-variable vector.
    Y : np.ndarray
        Binary label vector.
    results_dir : Path
        Directory in which the results CSV file is stored.
    k_values : list[int]
        Target dimensions used in the reconstruction experiment.
    experiment : str
        Experiment type. If ``"standard"``, the full dataset is used. If
        ``"equal_weights"``, the protected and unprotected group are sampled
        with equal weight.
    run_standard_pca : bool, optional
        Whether to run Standard PCA. The default is True.
    run_fair_pca : bool, optional
        Whether to run Fair PCA. The default is True.
    run_laftr : bool, optional
        Whether to run LAFTR. The default is True.
    """
    if not any([run_standard_pca, run_fair_pca, run_laftr]):
        raise ValueError("At least one reconstruction method must be selected.")

    # Select output file based on experiment type.
    if experiment == "standard":
        output_name = f"{dataset_name.lower()}_results_reconstruction.csv"
    elif experiment == "equal_weights":
        output_name = f"{dataset_name.lower()}_results_reconstruction_equal_weights.csv"
    else:
        raise ValueError("experiment must be either 'standard' or 'equal_weights'.")

    results_path = results_dir / output_name

    results_df = initialize_reconstruction_results(
        results_path=results_path,
        k_values=k_values,
    )

    print(f"\nRunning reconstruction experiment for {dataset_name}")

    if run_standard_pca:
        standard_results = run_standard_PCA(
            X=X,
            A=A,
            k_list=k_values,
        )

        (
            k_standard,
            error_0_standard,
            error_1_standard,
            loss_0_standard,
            loss_1_standard,
        ) = standard_results

        results_df = store_reconstruction_results(
            results_df=results_df,
            method_name="standard_PCA",
            k_list=k_standard,
            avg_errors_0=error_0_standard,
            avg_errors_1=error_1_standard,
            avg_losses_0=loss_0_standard,
            avg_losses_1=loss_1_standard,
        )

    if run_fair_pca:
        fair_results = run_fair_PCA(
            X=X,
            A=A,
            k_list=k_values,
        )

        (
            k_fair,
            error_0_fair,
            error_1_fair,
            loss_0_fair,
            loss_1_fair,
            rank_fair,
        ) = fair_results

        results_df = store_reconstruction_results(
            results_df=results_df,
            method_name="fair_PCA",
            k_list=k_fair,
            avg_errors_0=error_0_fair,
            avg_errors_1=error_1_fair,
            avg_losses_0=loss_0_fair,
            avg_losses_1=loss_1_fair,
            rank_values=rank_fair,
        )

    if run_laftr:
        laftr_results = run_LAFTR(
            X=X,
            A=A,
            Y=Y,
            k_list=k_values,
            n_epochs=20,
            batch_size=64,
            learning_rate=1e-3,
            class_coeff=1.0,
            recon_coeff=1.0,
            fair_coeff=1.0,
            center=True,
            scale=False,
            hidden_dim=32,
            patience=10,
            aud_steps=1,
            random_state=42,
            activation="leakyrelu",
            verbose=False,
        )

        (
            k_laftr,
            error_0_laftr,
            error_1_laftr,
            loss_0_laftr,
            loss_1_laftr,
            _,
        ) = laftr_results

        results_df = store_reconstruction_results(
            results_df=results_df,
            method_name="LAFTR",
            k_list=k_laftr,
            avg_errors_0=error_0_laftr,
            avg_errors_1=error_1_laftr,
            avg_losses_0=loss_0_laftr,
            avg_losses_1=loss_1_laftr,
        )

    results_df.to_csv(results_path, index=False)
    print(f"{dataset_name} results saved to: {results_path}")

# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================
def parse_args():
    """
    Parse command-line arguments for the reconstruction experiment script.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing the selected dataset and
        experiment type.
    """
    parser = argparse.ArgumentParser(
        description="Run reconstruction experiments for COMPAS, Adult, or both datasets."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["COMPAS", "Adult", "both"],
        default="both",
        help="Dataset to run. Use COMPAS, Adult, or both. Default is both.",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        choices=["standard", "equal_weights"],
        default="standard",
        help="Experiment type to run. Use standard or equal_weights. Default is standard.",
    )

    return parser.parse_args()

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    args = parse_args()

    RUN_STANDARD_PCA = True
    RUN_FAIR_PCA = True
    RUN_LAFTR = True

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_run = []

    if args.dataset in ["COMPAS", "both"]:
        if args.experiment == "standard":
            X_COMPAS, A_COMPAS, Y_COMPAS = wrap_COMPAS_full()
        else:
            X_COMPAS, A_COMPAS, Y_COMPAS = wrap_COMPAS_equal_weights()

        datasets_to_run.append(
            (
                "compas",
                X_COMPAS,
                A_COMPAS,
                Y_COMPAS,
                list(range(1, 11)),
            )
        )

    if args.dataset in ["Adult", "both"]:
        if args.experiment == "standard":
            X_Adult, A_Adult, Y_Adult = wrap_Adult_full()
        else:
            X_Adult, A_Adult, Y_Adult = wrap_Adult_equal_weights()

        datasets_to_run.append(
            (
                "adult",
                X_Adult,
                A_Adult,
                Y_Adult,
                list(range(1, 21)),
            )
        )

    for dataset_name, X, A, Y, k_values in datasets_to_run:
        run_reconstruction_experiment(
            dataset_name=dataset_name,
            X=X,
            A=A,
            Y=Y,
            results_dir=results_dir,
            k_values=k_values,
            experiment=args.experiment,
            run_standard_pca=RUN_STANDARD_PCA,
            run_fair_pca=RUN_FAIR_PCA,
            run_laftr=RUN_LAFTR,
        )
