"""
Run classification experiments for raw data, Standard PCA, Fair PCA, and LAFTR.

This module trains downstream classifiers across datasets and preprocessing
methods, selects representation dimensions by cross-validation, and stores
test-set predictions and CV losses for later evaluation.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import time
from typing import Any, Iterable, Callable

import numpy as np
import pandas as pd

# Surpress info and warning messages from TensorFlow.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from preprocessing.standard_pca import StandardPCA
from preprocessing.fair_pca import FairPCA
from preprocessing.laftr.laftr import LAFTR

from classifiers.log_reg import LogisticRegressionClassifier
from classifiers.svm import SVMClassifier
from classifiers.mlp import MLPClassifier

from experiments.utils import binary_log_loss, create_folds, format_elapsed_time, combine_fair_pca_split_representation
from data.load_data import wrap_COMPAS_split, wrap_Adult_split

# =============================================================================
# LOGGING HELPER
# =============================================================================
def print_elapsed_time(dataset_name: str, classifier_name: str, method_name: str, start_time: float) -> None:
    """Print elapsed runtime for one dataset-classifier-preprocessing combination."""
    elapsed_time = format_elapsed_time(time.perf_counter() - start_time)
    print(
        f"{dataset_name} | {classifier_name} | {method_name} | "
        f"elapsed time: {elapsed_time}"
    )

# =============================================================================
# CLASSIFIER HELPERS
# =============================================================================
def make_classifier(classifier_name: str):
    """
    Create a classifier instance from a classifier name.

    Parameters
    ----------
    classifier_name : str
        Name of the classifier to initialize. Supported values are "log_reg",
        "svm", and "mlp".

    Returns
    -------
    object
        Initialized classifier instance corresponding to ``classifier_name``.

    Raises
    ------
    ValueError
        If ``classifier_name`` is not one of the supported classifier names.
    """
    if classifier_name == "log_reg":
        return LogisticRegressionClassifier()

    if classifier_name == "svm":
        return SVMClassifier()

    if classifier_name == "mlp":
        return MLPClassifier()

    raise ValueError(f"Unknown classifier: {classifier_name}")
    
def train_predict(
    classifier_name: str,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train a selected classifier and return test-set predictions.

    Parameters
    ----------
    classifier_name : str
        Name of the classifier to train.
    X_train : np.ndarray
        Training feature matrix.
    Y_train : np.ndarray
        Training label vector.
    X_test : np.ndarray
        Test feature matrix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Predicted scores and predicted binary labels for ``X_test``.
    """
    classifier = make_classifier(classifier_name)
    classifier.train(X_train, Y_train)

    scores = classifier.predict_scores(X_test)
    labels = classifier.predict_labels(X_test)

    return scores, labels

# =============================================================================
# VALIDATION HELPERS
# =============================================================================
def validate_k_values(k_values: Iterable[int], n_features: int) -> list[int]:
    """
    Validate candidate representation dimensions.

    Parameters
    ----------
    k_values : Iterable[int]
        Candidate values for the reduced representation dimension.
    n_features : int
        Number of features in the original feature matrix.

    Returns
    -------
    list[int]
        Validated candidate k values as integers.

    Raises
    ------
    ValueError
        If no k values are provided, or if any k value does not satisfy
        ``0 < k < n_features``.
    """
    k_list = [int(k) for k in k_values]

    if not k_list:
        raise ValueError("k_values may not be empty.")

    invalid_k = [k for k in k_list if k <= 0 or k >= n_features]
    if invalid_k:
        raise ValueError(
            "All k values must satisfy 0 < k < number of features. "
            f"Invalid k values: {invalid_k}."
        )

    return k_list

def validate_finite_array(array: np.ndarray, name: str) -> None:
    """
    Check that an array contains only finite values.

    Parameters
    ----------
    array : np.ndarray
        Array to validate.
    name : str
        Name used to identify the array in the error message.

    Raises
    ------
    ValueError
        If the array contains NaN or infinite values.
    """
    array = np.asarray(array)

    if not np.all(np.isfinite(array)):
        n_nan = int(np.isnan(array).sum())
        n_inf = int(np.isinf(array).sum())
        raise ValueError(
            f"{name} contains non-finite values "
            f"(NaN: {n_nan}, infinity: {n_inf})."
        )

# =============================================================================
# PREPROCESSING METHOD RUNNERS
# =============================================================================
# NO PREPROCESSING
# -----------------------------------------------------------------------------
def run_raw_data_baseline(
    classifier_name: str,
    X_train: np.ndarray,
    A_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    A_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train the selected classifier on the original feature matrix.

    This function represents the no-preprocessing baseline. The sensitive-variable
    vectors are included for consistency with the other preprocessing runners but
    are not used by the classifier.

    Parameters
    ----------
    classifier_name : str
        Name of the classifier to train.
    X_train : np.ndarray
        Original training feature matrix.
    A_train : np.ndarray
        Training sensitive-variable vector.
    Y_train : np.ndarray
        Training label vector.
    X_test : np.ndarray
        Original test feature matrix.
    A_test : np.ndarray
        Test sensitive-variable vector.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Predicted scores and predicted binary labels for ``X_test``.
    """
    return train_predict(classifier_name, X_train, Y_train, X_test)

# -----------------------------------------------------------------------------
# STANDARD PCA PREPROCESSING
# -----------------------------------------------------------------------------
def run_standard_PCA(
    classifier_name: str,
    k: int,
    X_train: np.ndarray,
    A_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    A_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run classification after Standard PCA preprocessing.

    Standard PCA is fitted on the training feature matrix and then used to
    transform both the training and test feature matrices. The selected classifier
    is trained on the reduced training representation and evaluated on the reduced
    test representation. The sensitive-variable vectors are included for interface
    consistency but are not used by Standard PCA.

    Parameters
    ----------
    classifier_name : str
        Name of the classifier to train.
    k : int
        Number of PCA dimensions.
    X_train : np.ndarray
        Training feature matrix.
    A_train : np.ndarray
        Training sensitive-variable vector.
    Y_train : np.ndarray
        Training label vector.
    X_test : np.ndarray
        Test feature matrix.
    A_test : np.ndarray
        Test sensitive-variable vector.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Predicted scores and predicted binary labels for the reduced test data.
    """
    preprocessor = StandardPCA(k)
    preprocessor.fit(X_train)

    Z_train = preprocessor.transform(X_train)
    Z_test = preprocessor.transform(X_test)

    scores, labels = train_predict(classifier_name, Z_train, Y_train, Z_test)

    return scores, labels

# -----------------------------------------------------------------------------
# FAIR PCA PREPROCESSING
# -----------------------------------------------------------------------------
def run_fair_PCA(
    classifier_name: str,
    k: int,
    X_train: np.ndarray,
    A_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    A_test: np.ndarray,
) -> tuple[FairPCA, np.ndarray, np.ndarray]:
    """
    Run classification after Fair PCA preprocessing.

    Fair PCA is fitted on the training feature matrix using the training
    sensitive-variable vector. Since the Fair PCA transform method returns
    separate representations for the unprotected and protected groups, these
    group-specific representations are recombined into the original row order
    before training and testing the selected classifier.

    Parameters
    ----------
    classifier_name : str
        Name of the classifier to train.
    k : int
        Target dimension used by Fair PCA.
    X_train : np.ndarray
        Training feature matrix.
    A_train : np.ndarray
        Training sensitive-variable vector.
    Y_train : np.ndarray
        Training label vector.
    X_test : np.ndarray
        Test feature matrix.
    A_test : np.ndarray
        Test sensitive-variable vector.

    Returns
    -------
    tuple[FairPCA, np.ndarray, np.ndarray]
        Fitted Fair PCA preprocessor, predicted scores, and predicted binary
        labels for the reduced test data.
    """
    preprocessor = FairPCA(k)
    preprocessor.fit(X_train, A_train)

    Z0_train, Z1_train = preprocessor.transform(X_train, A_train)
    Z0_test, Z1_test = preprocessor.transform(X_test, A_test)

    Z_train = combine_fair_pca_split_representation(Z0_train, Z1_train, A_train)
    Z_test = combine_fair_pca_split_representation(Z0_test, Z1_test, A_test)

    scores, labels = train_predict(classifier_name, Z_train, Y_train, Z_test)

    return preprocessor, scores, labels

# -----------------------------------------------------------------------------
# LAFTR PREPROCESSING
# -----------------------------------------------------------------------------
def run_LAFTR(
    classifier_name: str,
    k: int,
    X_train: np.ndarray,
    A_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    A_test: np.ndarray,
    **laftr_kwargs: Any,
) -> tuple[LAFTR, np.ndarray, np.ndarray]:
    """
    Run classification after LAFTR preprocessing.

    LAFTR is fitted on the training feature matrix, sensitive-variable vector,
    and label vector. The learned representation is then used as input for the
    selected downstream classifier. The internal LAFTR classifier is not used for
    the final prediction step.

    Parameters
    ----------
    classifier_name : str
        Name of the downstream classifier to train.
    k : int
        Dimension of the learned LAFTR representation.
    X_train : np.ndarray
        Training feature matrix.
    A_train : np.ndarray
        Training sensitive-variable vector.
    Y_train : np.ndarray
        Training label vector.
    X_test : np.ndarray
        Test feature matrix.
    A_test : np.ndarray
        Test sensitive-variable vector.
    **laftr_kwargs : Any
        Additional keyword arguments passed to the ``LAFTR`` constructor.

    Returns
    -------
    tuple[LAFTR, np.ndarray, np.ndarray]
        Fitted LAFTR preprocessor, predicted scores, and predicted binary labels
        for the reduced test data.
    """
    preprocessor = LAFTR(k=k, **laftr_kwargs)

    try:
        preprocessor.fit(X_train, A_train, Y_train)

        Z_train = preprocessor.transform(X_train)
        Z_test = preprocessor.transform(X_test)

        validate_finite_array(Z_train, f"LAFTR Z_train for k={k}")
        validate_finite_array(Z_test, f"LAFTR Z_test for k={k}")

    finally:
        preprocessor.close()

    scores, labels = train_predict(classifier_name, Z_train, Y_train, Z_test)

    return preprocessor, scores, labels

# =============================================================================
# CROSS-VALIDATION HELPERS
# =============================================================================
def cross_validate_k(
    method_name: str,
    classifier_name: str,
    X_train: np.ndarray,
    A_train: np.ndarray,
    Y_train: np.ndarray,
    k_values: Iterable[int],
    run_method: Callable[..., tuple[np.ndarray, np.ndarray] | tuple[Any, np.ndarray, np.ndarray]],
    min_k: int = 1,
    **method_kwargs: Any,
) -> tuple[int, float, dict[str, list[str | int | float]]]:
    """
    Select the representation dimension by cross-validated downstream log-loss.

    For each candidate value of k, the preprocessing method is fitted on each
    training fold and evaluated on the corresponding validation fold. The final
    value of k is selected using the one-standard-error rule.

    Parameters
    ----------
    method_name : str
        Name of the preprocessing method used in printed progress messages and
        stored CV results.
    classifier_name : str
        Name of the downstream classifier used in printed progress messages and
        stored CV results.
    X_train : np.ndarray
        Training feature matrix.
    A_train : np.ndarray
        Training sensitive-variable vector.
    Y_train : np.ndarray
        Training label vector.
    k_values : Iterable[int]
        Candidate values for the reduced representation dimension.
    run_method : Callable
        Function that runs the preprocessing method for one fixed value of k.
        The function must return either ``(scores, labels)`` or
        ``(preprocessor, scores, labels)``.
    min_k : int, optional
        Minimum allowed value of k after validation. The default is 1.
    **method_kwargs : Any
        Additional keyword arguments passed to ``run_method``.

    Returns
    -------
    tuple[int, float, dict[str, list[str | int | float]]]
        Selected k, selected-k average log-loss, and a CV-results dictionary
        containing the classifier name, preprocessing method, k values, average
        validation log-losses, and standard errors of the fold-level
        validation log-losses.

    Raises
    ------
    ValueError
        If no valid k values remain after applying ``min_k``.
    """
    # Create folds for cross-validation.
    folds = create_folds(X_train, A_train, Y_train)
    
    # Initialise list of target dimension-values.
    _, n_features = X_train.shape
    k_list = validate_k_values(k_values, n_features)
    k_list = [k for k in k_list if k >= min_k]

    if not k_list:
        raise ValueError(f"{method_name} requires at least one candidate k >= {min_k}.")

    # Initialise dictionaries.
    avg_log_losses: dict[int, float] = {}
    fold_log_losses_dict: dict[int, list[float]] = {}

    cv_results: dict[str, list[str | int | float]] = {
        "classifier": [],
        "preprocessing": [],
        "k": [],
        "avg_log_loss": [],
        "standard_error": [],
    }
    
    # Iterate over values of k.
    for k in k_list:
        fold_log_losses = []

        # Iterate over folds.
        for (X_train_fold, A_train_fold, Y_train_fold,
             X_val_fold, A_val_fold, Y_val_fold) in folds:
            result = run_method(
                classifier_name,
                k,
                X_train_fold,
                A_train_fold,
                Y_train_fold,
                X_val_fold,
                A_val_fold,
                **method_kwargs,
            )

            if len(result) == 2:
                scores_pred_fold, _ = result
            else:
                _, scores_pred_fold, _ = result

            # Compute binary log loss for this validation fold.
            log_loss_fold = binary_log_loss(Y_val_fold, scores_pred_fold)
            fold_log_losses.append(log_loss_fold)
            
        # Compute average log loss and standard error for this value of k.
        avg_log_loss = float(np.mean(fold_log_losses))
        standard_error = compute_standard_error(fold_log_losses)
        
        # Update dictionaries.
        avg_log_losses[k] = avg_log_loss
        fold_log_losses_dict[k] = fold_log_losses
        
        cv_results["classifier"].append(classifier_name)
        cv_results["preprocessing"].append(method_name)
        cv_results["k"].append(k)
        cv_results["avg_log_loss"].append(avg_log_loss)
        cv_results["standard_error"].append(standard_error)

        print(f"{classifier_name} | {method_name} | k={k} finished")
        
    # Select best value of k with one standard error rule.
    best_k, best_k_loss, min_loss_k, min_loss = select_k_one_standard_error(
        avg_log_losses,
        fold_log_losses_dict,
    )

    print(
        f"{classifier_name} | {method_name} | minimum-loss k: {min_loss_k} "
        f"({min_loss:.6f}) | selected k: {best_k} ({best_k_loss:.6f})"
    )

    return best_k, best_k_loss, cv_results

def compute_standard_error(values: list[float]) -> float:
    """
    Compute the standard error of fold-level validation losses.

    Parameters
    ----------
    values : list[float]
        Fold-level validation losses for one candidate k.

    Returns
    -------
    float
        Standard error across folds. If only one fold is available, the
        standard error is set to 0.0.
    """
    values_array = np.asarray(values, dtype=float)

    if len(values_array) <= 1:
        return 0.0

    return float(np.std(values_array, ddof=1) / np.sqrt(len(values_array)))

def select_k_one_standard_error(
    avg_log_losses: dict[int, float],
    fold_log_losses_dict: dict[int, list[float]],
) -> tuple[int, float, int, float]:
    """
    Select the representation dimension using the one-standard-error rule.

    The minimum-loss k is first identified as the candidate with the lowest
    average validation log-loss. The one-standard-error threshold is then
    computed as this minimum average log-loss plus the standard error of the
    fold-level log-losses for the minimum-loss k. Among all candidate k values
    whose average log-loss is below this threshold, the smallest k is selected.

    Parameters
    ----------
    avg_log_losses : dict[int, float]
        Dictionary mapping each candidate k value to its average validation
        log-loss.
    fold_log_losses_dict : dict[int, list[float]]
        Dictionary mapping each candidate k value to the corresponding
        fold-level validation log-losses.

    Returns
    -------
    tuple[int, float, int, float]
        Selected k, average log-loss of the selected k, minimum-loss k, and
        average log-loss of the minimum-loss k.

    Raises
    ------
    ValueError
        If ``avg_log_losses`` is empty.
    """
    if not avg_log_losses:
        raise ValueError("avg_log_losses must not be empty.")

    min_loss_k = min(avg_log_losses, key=avg_log_losses.get)
    min_loss = avg_log_losses[min_loss_k]

    min_loss_standard_error = compute_standard_error(
        fold_log_losses_dict[min_loss_k]
    )

    one_se_threshold = min_loss + min_loss_standard_error

    eligible_k = [
        k for k, avg_loss in avg_log_losses.items()
        if avg_loss <= one_se_threshold
    ]

    best_k = min(eligible_k)
    best_k_loss = avg_log_losses[best_k]

    return best_k, best_k_loss, min_loss_k, min_loss

# =============================================================================
# RESULT EXPORT HELPERS
# =============================================================================
def store_method_results(
    results_df: pd.DataFrame,
    method_name: str,
    scores: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Store method-specific scores and labels in the results dataframe.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing at least Y_test and A_test.
    method_name : str
        Name of the preprocessing method.
    scores : np.ndarray
        Predicted positive-class scores.
    labels : np.ndarray
        Predicted binary labels.

    Returns
    -------
    pd.DataFrame
        Updated results dataframe.
    """
    if len(scores) != len(results_df) or len(labels) != len(results_df):
        raise ValueError(
            f"{method_name} produced a different number of predictions than the test-set size."
        )

    results_df[f"Y_score_{method_name}"] = np.asarray(scores, dtype=float)
    results_df[f"Y_pred_{method_name}"] = np.asarray(labels, dtype=int)

    return results_df

def store_CV_results(
    cv_results_df: pd.DataFrame,
    method_name: str,
    cv_results: dict[str, list[str | int | float]],
) -> pd.DataFrame:
    """
    Store cross-validation losses and standard errors for one preprocessing method.

    Parameters
    ----------
    cv_results_df : pd.DataFrame
        Existing CV results dataframe. It should contain one row per k.
    method_name : str
        Name of the preprocessing method.
    cv_results : dict[str, list[str | int | float]]
        Cross-validation results returned by ``cross_validate_k``.

    Returns
    -------
    pd.DataFrame
        Updated CV results dataframe with one average-loss column and one
        standard-error column added or replaced for ``method_name``.
    """
    column_map = {
        "standard_PCA": ("standard_PCA_avg_loss", "standard_PCA_standard_error"),
        "fair_PCA": ("fair_PCA_avg_loss", "fair_PCA_standard_error"),
        "LAFTR": ("LAFTR_avg_loss", "LAFTR_standard_error"),
    }

    if method_name not in column_map:
        raise ValueError(f"Unknown CV method name: {method_name}")

    loss_column, standard_error_column = column_map[method_name]

    method_cv_df = pd.DataFrame(
        {
            "k": cv_results["k"],
            loss_column: cv_results["avg_log_loss"],
            standard_error_column: cv_results["standard_error"],
        }
    )

    if cv_results_df.empty:
        return method_cv_df.sort_values("k").reset_index(drop=True)

    columns_to_replace = [
        column for column in [loss_column, standard_error_column]
        if column in cv_results_df.columns
    ]
    if columns_to_replace:
        cv_results_df = cv_results_df.drop(columns=columns_to_replace)

    cv_results_df = cv_results_df.merge(method_cv_df, on="k", how="outer")
    cv_results_df = cv_results_df.sort_values("k").reset_index(drop=True)

    return cv_results_df

# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================
def parse_args():
    """
    Parse command-line arguments for the classification experiment script.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing the selected dataset and 
        classifier.
    """
    parser = argparse.ArgumentParser(
        description="Run classification experiments for selected datasets and classifiers."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["COMPAS", "Adult", "both"],
        default="both",
        help="Dataset to run. Use COMPAS, Adult, or both. Default is both.",
    )

    parser.add_argument(
        "--classifier",
        type=str,
        choices=["log_reg", "svm", "mlp", "all"],
        default="all",
        help="Classifier to run. Use log_reg, svm, mlp, or all. Default is all.",
    )

    return parser.parse_args()

def get_datasets_to_run(dataset_arg: str):
    """
    Load the selected datasets.

    Parameters
    ----------
    dataset_arg : str
        Dataset argument from parse_args: COMPAS, Adult, or both.

    Returns
    -------
    list[tuple]
        Tuples containing dataset name and train/test arrays.
    """
    datasets_to_run = []

    if dataset_arg in ["COMPAS", "both"]:
        X_train, A_train, Y_train, X_test, A_test, Y_test = wrap_COMPAS_split()
        datasets_to_run.append(
            ("compas", X_train, A_train, Y_train, X_test, A_test, Y_test)
        )

    if dataset_arg in ["Adult", "both"]:
        X_train, A_train, Y_train, X_test, A_test, Y_test = wrap_Adult_split()
        datasets_to_run.append(
            ("adult", X_train, A_train, Y_train, X_test, A_test, Y_test)
        )

    return datasets_to_run

def get_classifiers_to_run(classifier_arg: str) -> list[str]:
    """
    Convert the classifier command-line argument into a list of classifier names.

    Parameters
    ----------
    classifier_arg : str
        Classifier argument supplied through the command line. If set to
        "all", all supported classifiers are returned. Otherwise, the
        argument is treated as the name of a single classifier.

    Returns
    -------
    list[str]
        List of classifier names to run. The full list is returned for
        classifier_arg="all"; otherwise, a one-element list containing
        classifier_arg is returned.
    """
    if classifier_arg == "all":
        return ["log_reg", "svm", "mlp"]

    return [classifier_arg]

def get_k_values(dataset_name: str) -> list[int]:
    """
    Return the candidate k values used for cross-validation.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    list[int]
        Dataset-specific candidate representation dimensions.
    """
    if dataset_name == "compas":
        return list(range(1, 11))

    if dataset_name == "adult":
        return list(range(2, 41, 2))

    raise ValueError(f"Unknown dataset name: {dataset_name}")

# =============================================================================
# MAIN EXPERIMENT WRAPPER
# =============================================================================
def run_classification_experiment(
    dataset_name: str,
    classifier_name: str,
    X_train: np.ndarray,
    A_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    A_test: np.ndarray,
    Y_test: np.ndarray,
    results_dir: Path,
    k_values: list[int],
    run_raw_data: bool = True,
    run_standard_pca: bool = True,
    run_fair_pca: bool = True,
    run_laftr: bool = True,
    **laftr_kwargs: Any,
) -> None:
    """
    Run classification experiments for one dataset-classifier combination.

    The function optionally runs the raw-data baseline, Standard PCA, Fair PCA,
    and LAFTR. For the dimensionality-reducing preprocessing methods, the target
    dimension k is selected by cross-validation on the training set. Final
    predictions are then generated on the test set using the selected k.

    Test-set predictions are stored in the classification-results CSV. The
    cross-validation losses for each candidate k are stored in a separate CV
    results CSV.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset, used in the output filenames.
    classifier_name : str
        Name of the classifier: log_reg, svm, or mlp.
    X_train : np.ndarray
        Training feature matrix.
    A_train : np.ndarray
        Training sensitive-variable vector.
    Y_train : np.ndarray
        Training label vector.
    X_test : np.ndarray
        Test feature matrix.
    A_test : np.ndarray
        Test sensitive-variable vector.
    Y_test : np.ndarray
        Test label vector.
    results_dir : Path
        Directory in which the CSV files are saved.
    k_values : list[int]
        Candidate dimensions for cross-validation.
    run_raw_data : bool, optional
        Whether to run the raw-data baseline. The default is True.
    run_standard_pca : bool, optional
        Whether to run Standard PCA preprocessing. The default is True.
    run_fair_pca : bool, optional
        Whether to run Fair PCA preprocessing. The default is True.
    run_laftr : bool, optional
        Whether to run LAFTR preprocessing. The default is True.
    **laftr_kwargs : Any
        Keyword arguments passed to LAFTR.
    """
    if not any([run_raw_data, run_standard_pca, run_fair_pca, run_laftr]):
        raise ValueError("At least one classification method must be selected.")

    
    # Set output paths.
    output_name = f"{dataset_name}_results_classification_{classifier_name}.csv"
    results_path = results_dir / output_name

    cv_output_name = f"{dataset_name}_cv_results_{classifier_name}.csv"
    cv_results_path = results_dir / cv_output_name

    # Initialize or load test-results dataframe.
    if results_path.exists():
        results_df = pd.read_csv(results_path)

        if len(results_df) != len(Y_test):
            raise ValueError(
                "Existing results file has a different number of rows than the "
                "current test set. Delete the old file or use a different output "
                "filename."
            )

        required_columns = {"Y_test", "A_test"}
        if not required_columns.issubset(results_df.columns):
            raise ValueError(
                "Existing results file does not contain the required Y_test and "
                "A_test columns. Delete the old file or use a different output "
                "filename."
            )
    else:
        results_df = pd.DataFrame(
            {
                "Y_test": np.asarray(Y_test, dtype=int),
                "A_test": np.asarray(A_test, dtype=int),
            }
        )

    # Initialize or load CV-results dataframe
    if cv_results_path.exists():
        cv_results_df = pd.read_csv(cv_results_path)

        if "k" not in cv_results_df.columns:
            raise ValueError(
                "Existing CV results file does not contain a k column. Delete "
                "the old file or use a different output filename."
            )
    else:
        cv_results_df = pd.DataFrame()

    print(f"\nRunning {dataset_name} | {classifier_name}")

    # Run raw-data baseline
    if run_raw_data:
        method_start_time = time.perf_counter()
        print(f"{dataset_name} | {classifier_name} | raw data")

        scores_raw, labels_raw = run_raw_data_baseline(
            classifier_name,
            X_train,
            A_train,
            Y_train,
            X_test,
            A_test,
        )

        results_df = store_method_results(
            results_df,
            "raw_data",
            scores_raw,
            labels_raw,
        )

        print_elapsed_time(
            dataset_name,
            classifier_name,
            "raw data",
            method_start_time,
        )

    # Run Standard PCA
    if run_standard_pca:
        method_start_time = time.perf_counter()
        print(f"{dataset_name} | {classifier_name} | Standard PCA")

        best_k_standard, _, cv_standard = cross_validate_k(
            method_name="standard_PCA",
            classifier_name=classifier_name,
            X_train=X_train,
            A_train=A_train,
            Y_train=Y_train,
            k_values=k_values,
            run_method=run_standard_PCA,
        )

        cv_results_df = store_CV_results(
            cv_results_df,
            "standard_PCA",
            cv_standard,
        )

        scores_standard_pca, labels_standard_pca = run_standard_PCA(
            classifier_name,
            best_k_standard,
            X_train,
            A_train,
            Y_train,
            X_test,
            A_test,
        )

        results_df = store_method_results(
            results_df,
            "standard_PCA",
            scores_standard_pca,
            labels_standard_pca,
        )

        print_elapsed_time(
            dataset_name,
            classifier_name,
            "Standard PCA",
            method_start_time,
        )

    # Run Fair PCA
    if run_fair_pca:
        method_start_time = time.perf_counter()
        print(f"{dataset_name} | {classifier_name} | Fair PCA")

        best_k_fair, _, cv_fair = cross_validate_k(
            method_name="fair_PCA",
            classifier_name=classifier_name,
            X_train=X_train,
            A_train=A_train,
            Y_train=Y_train,
            k_values=k_values,
            run_method=run_fair_PCA,
        )

        cv_results_df = store_CV_results(
            cv_results_df,
            "fair_PCA",
            cv_fair,
        )

        _, scores_fair_pca, labels_fair_pca = run_fair_PCA(
            classifier_name,
            best_k_fair,
            X_train,
            A_train,
            Y_train,
            X_test,
            A_test,
        )

        results_df = store_method_results(
            results_df,
            "fair_PCA",
            scores_fair_pca,
            labels_fair_pca,
        )

        print_elapsed_time(
            dataset_name,
            classifier_name,
            "Fair PCA",
            method_start_time,
        )

    # Run LAFTR
    if run_laftr:
        method_start_time = time.perf_counter()
        print(f"{dataset_name} | {classifier_name} | LAFTR")

        # Exclude k=1 for LAFTR due to empirical instabilities.
        best_k_laftr, _, cv_laftr = cross_validate_k(
            method_name="LAFTR",
            classifier_name=classifier_name,
            X_train=X_train,
            A_train=A_train,
            Y_train=Y_train,
            k_values=k_values,
            run_method=run_LAFTR,
            min_k=2,
            **laftr_kwargs,
        )

        cv_results_df = store_CV_results(
            cv_results_df,
            "LAFTR",
            cv_laftr,
        )

        _, scores_laftr, labels_laftr = run_LAFTR(
            classifier_name,
            best_k_laftr,
            X_train,
            A_train,
            Y_train,
            X_test,
            A_test,
            **laftr_kwargs,
        )

        results_df = store_method_results(
            results_df,
            "LAFTR",
            scores_laftr,
            labels_laftr,
        )

        print_elapsed_time(
            dataset_name,
            classifier_name,
            "LAFTR",
            method_start_time,
        )

    # Export results
    results_df.to_csv(results_path, index=False)

    if not cv_results_df.empty:
        cv_results_df.to_csv(cv_results_path, index=False)

    print(f"Results saved to: {results_path}")

    if not cv_results_df.empty:
        print(f"CV results saved to: {cv_results_path}")
        
# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    args = parse_args()

    RUN_RAW_DATA = True
    RUN_STANDARD_PCA = True
    RUN_FAIR_PCA = True
    RUN_LAFTR = True

    LAFTR_KWARGS = {
        "center": True,
        "scale": False,
        "class_coeff": 1.0,
        "recon_coeff": 1.0,
        "fair_coeff": 1.0,
        "hidden_dim": 32,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "n_epochs": 100,
        "patience": 10,
        "aud_steps": 1,
        "random_state": 42,
        "activation": "leakyrelu",
        "verbose": False,
    }

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_run = get_datasets_to_run(args.dataset)
    classifiers_to_run = get_classifiers_to_run(args.classifier)

    for (
        dataset_name,
        X_train,
        A_train,
        Y_train,
        X_test,
        A_test,
        Y_test,
    ) in datasets_to_run:
        for classifier_name in classifiers_to_run:
            run_classification_experiment(
                dataset_name=dataset_name,
                classifier_name=classifier_name,
                X_train=X_train,
                A_train=A_train,
                Y_train=Y_train,
                X_test=X_test,
                A_test=A_test,
                Y_test=Y_test,
                results_dir=results_dir,
                k_values=get_k_values(dataset_name),
                run_raw_data=RUN_RAW_DATA,
                run_standard_pca=RUN_STANDARD_PCA,
                run_fair_pca=RUN_FAIR_PCA,
                run_laftr=RUN_LAFTR,
                **LAFTR_KWARGS,
            )