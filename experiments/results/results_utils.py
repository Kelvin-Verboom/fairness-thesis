"""
Utility functions for printing, plotting, and saving results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from evaluation.prediction import accuracy, accuracy_protected, disparity, opty, odds

PREPROCESSING_METHODS = {
    "standard_PCA": "Standard PCA",
    "fair_PCA": "Fair PCA",
    "LAFTR": "LAFTR",
}

CLASSIFIERS = {
    "log_reg": "Logistic Regression",
    "svm": "SVM",
    "mlp": "MLP",
}

DATASETS = {
    "compas": "COMPAS",
    "adult": "Adult",
}

# =============================================================================
# DATASETS SUMMARY HELPER
# =============================================================================
def summarize_prepared_sample(
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray
    ) -> dict[str, str]:
    """
    Compute formatted summary statistics for one prepared sample.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    A : np.ndarray
        Binary sensitive-variable vector, where 1 indicates the protected group.
    Y : np.ndarray
        Binary label vector, where 1 indicates the positive label.

    Returns
    -------
    dict[str, str]
        Formatted summary statistics for the prepared sample.
    """
    n_observations, n_features = X.shape

    protected_count = int(A.sum())
    positive_label_count = int(Y.sum())

    protected_share = 100 * protected_count / n_observations
    positive_label_share = 100 * positive_label_count / n_observations

    return {
        "Observations": f"{n_observations:,}",
        "Features": f"{n_features:,}",
        "Protected group": f"{protected_count:,} ({protected_share:.2f}%)",
        "Positive label": f"{positive_label_count:,} ({positive_label_share:.2f}%)",
    }

def print_prepared_dataset_summary(
        dataset_name: str,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray
    ) -> None:
    """
    Print summary statistics for a prepared dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    X : np.ndarray
        Prepared feature matrix.
    A : np.ndarray
        Binary sensitive-variable vector, where 1 indicates the protected group.
    Y : np.ndarray
        Binary label vector, where 1 indicates the positive label.
    """
    summary = summarize_prepared_sample(X, A, Y)

    print(f"\n{dataset_name} prepared dataset summary")
    print("----------------------------------------")
    print(f"Observations:    {summary['Observations']}")
    print(f"Features:        {summary['Features']}")
    print(f"Protected group: {summary['Protected group']}")
    print(f"Positive label:  {summary['Positive label']}")
    
def print_prepared_split_dataset_summary(
        dataset_name: str,
        X_train: np.ndarray,
        A_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        A_test: np.ndarray,
        Y_test: np.ndarray,
    ) -> None:
    """
    Print summary statistics for a prepared train-test split dataset.

    The function reports the number of observations, number of features,
    protected-group count and share, and positive-label count and share for
    the training sample, testing sample, and full sample.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    X_train : np.ndarray
        Training feature matrix.
    A_train : np.ndarray
        Training binary sensitive-variable vector, where 1 indicates the
        protected group.
    Y_train : np.ndarray
        Training binary label vector, where 1 indicates the positive label.
    X_test : np.ndarray
        Testing feature matrix.
    A_test : np.ndarray
        Testing binary sensitive-variable vector, where 1 indicates the
        protected group.
    Y_test : np.ndarray
        Testing binary label vector, where 1 indicates the positive label.
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_full = n_train + n_test

    n_features = X_train.shape[1]

    protected_train = int(A_train.sum())
    protected_test = int(A_test.sum())
    protected_full = protected_train + protected_test

    positive_train = int(Y_train.sum())
    positive_test = int(Y_test.sum())
    positive_full = positive_train + positive_test

    train_summary = {
        "Observations": f"{n_train:,}",
        "Features": f"{n_features:,}",
        "Protected group": f"{protected_train:,} ({100 * protected_train / n_train:.2f}%)",
        "Positive label": f"{positive_train:,} ({100 * positive_train / n_train:.2f}%)",
    }

    test_summary = {
        "Observations": f"{n_test:,}",
        "Features": f"{n_features:,}",
        "Protected group": f"{protected_test:,} ({100 * protected_test / n_test:.2f}%)",
        "Positive label": f"{positive_test:,} ({100 * positive_test / n_test:.2f}%)",
    }

    full_summary = {
        "Observations": f"{n_full:,}",
        "Features": f"{n_features:,}",
        "Protected group": f"{protected_full:,} ({100 * protected_full / n_full:.2f}%)",
        "Positive label": f"{positive_full:,} ({100 * positive_full / n_full:.2f}%)",
    }

    row_names = [
        "Observations",
        "Features",
        "Protected group",
        "Positive label",
    ]

    print(f"\n{dataset_name} prepared split dataset summary")
    print("-" * 78)
    print(
        f"{'':<18}"
        f"{'Training sample':>20}"
        f"{'Testing sample':>20}"
        f"{'Full sample':>20}"
    )
    print("-" * 78)

    for row in row_names:
        print(
            f"{row:<18}"
            f"{train_summary[row]:>20}"
            f"{test_summary[row]:>20}"
            f"{full_summary[row]:>20}"
        )
# =============================================================================
# LOADING RESULTS HELPER
# ============================================================================= 
def load_reconstruction_results(results_path: str | Path) -> pd.DataFrame:
    """
    Load reconstruction results from a CSV file.

    Parameters
    ----------
    results_path : str | Path
        Path to the reconstruction results CSV file.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the reconstruction results.
    """
    results_path = Path(results_path)
    results_df = pd.read_csv(results_path)

    required_columns = [
        "k",
        "avg_error_A0_standard_PCA",
        "avg_error_A1_standard_PCA",
        "avg_loss_A0_standard_PCA",
        "avg_loss_A1_standard_PCA",
        "avg_error_A0_fair_PCA",
        "avg_error_A1_fair_PCA",
        "avg_loss_A0_fair_PCA",
        "avg_loss_A1_fair_PCA",
        "rank_fair_PCA",
        "avg_error_A0_LAFTR",
        "avg_error_A1_LAFTR",
        "avg_loss_A0_LAFTR",
        "avg_loss_A1_LAFTR",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in results_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {results_path.name}: "
            f"{missing_columns}"
        )

    return results_df

def load_CV_results(results_path: str | Path) -> pd.DataFrame:
    """
    Load cross-validation results from a classification CV CSV file.

    Parameters
    ----------
    results_path : str | Path
        Path to the cross-validation results CSV file.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the cross-validation results, sorted by k.
    """
    results_path = Path(results_path)
    results_df = pd.read_csv(results_path)

    required_columns = [
        "k",
        "standard_PCA_avg_loss",
        "standard_PCA_standard_error",
        "fair_PCA_avg_loss",
        "fair_PCA_standard_error",
        "LAFTR_avg_loss",
        "LAFTR_standard_error",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in results_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {results_path.name}: "
            f"{missing_columns}"
        )

    results_df = results_df.copy()
    results_df["k"] = results_df["k"].astype(int)
    results_df = results_df.sort_values("k").reset_index(drop=True)

    return results_df

def load_classification_results(results_path: str | Path) -> pd.DataFrame:
    """
    Load classification results from a CSV file.

    Parameters
    ----------
    results_path : str | Path
        Path to the classification results CSV file.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the classification results.
    """
    results_path = Path(results_path)
    results_df = pd.read_csv(results_path)

    required_columns = [
        "Y_test",
        "A_test",
        "Y_score_raw_data",
        "Y_pred_raw_data",
        "Y_score_standard_PCA",
        "Y_pred_standard_PCA",
        "Y_score_fair_PCA",
        "Y_pred_fair_PCA",
        "Y_score_LAFTR",
        "Y_pred_LAFTR",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in results_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {results_path.name}: "
            f"{missing_columns}"
        )

    results_df = results_df.copy()

    binary_columns = [
        "Y_test",
        "A_test",
        "Y_pred_raw_data",
        "Y_pred_standard_PCA",
        "Y_pred_fair_PCA",
        "Y_pred_LAFTR",
    ]

    score_columns = [
        "Y_score_raw_data",
        "Y_score_standard_PCA",
        "Y_score_fair_PCA",
        "Y_score_LAFTR",
    ]

    for column in binary_columns:
        results_df[column] = results_df[column].astype(int)

    for column in score_columns:
        results_df[column] = results_df[column].astype(float)

    return results_df

# =============================================================================
# RECONSTRUCTION PLOTTING HELPERS
# ============================================================================= 
def get_reconstruction_metric_columns(
        method_key: str,
        metric_type: str
    ) -> tuple[str, str]:
    """
    Return the group-specific reconstruction metric columns.

    Parameters
    ----------
    method_key : str
        Method identifier. Must be one of "standard_PCA", "fair_PCA",
        or "LAFTR".
    metric_type : str
        Metric type. Must be either "error" or "loss".

    Returns
    -------
    tuple[str, str]
        Column names for the unprotected and protected group metrics.
    """
    if method_key not in PREPROCESSING_METHODS:
        raise ValueError(
            f"Unknown method_key '{method_key}'. "
            f"Expected one of {list(PREPROCESSING_METHODS)}."
        )

    if metric_type not in {"error", "loss"}:
        raise ValueError("metric_type must be either 'error' or 'loss'.")

    column_A0 = f"avg_{metric_type}_A0_{method_key}"
    column_A1 = f"avg_{metric_type}_A1_{method_key}"

    return column_A0, column_A1

def make_reconstruction_figure_name(
        dataset_name: str,
        method_key: str,
        metric_type: str,
        equal_weights: bool = False
    ) -> str:
    """
    Create a filename for a reconstruction plot.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    method_key : str
        Method identifier. Must be one of "standard_PCA", "fair_PCA",
        or "LAFTR".
    metric_type : str
        Metric type, either "error" or "loss".
    equal_weights : bool, optional
        Whether the results are based on equal group weights. The default is
        False.

    Returns
    -------
    str
        Figure filename.
    """
    suffix = "_equal_weights" if equal_weights else ""

    return (
        f"{dataset_name.lower()}_{method_key}_"
        f"reconstruction_{metric_type}{suffix}.png"
    ) 

def plot_single_reconstruction_metric(
        results_df: pd.DataFrame,
        dataset_name: str,
        method_key: str,
        metric_type: str,
        y_lim: tuple[float, float] | None = None,
        figures_dir: str | Path | None = None,
        equal_weights: bool = False
    ) -> None:
    """
    Plot one reconstruction metric for one preprocessing method.

    Parameters
    ----------
    results_df : pd.DataFrame
        Dataframe containing reconstruction results.
    dataset_name : str
        Name of the dataset shown in the plot title.
    method_key : str
        Method identifier. Must be one of "standard_PCA", "fair_PCA",
        or "LAFTR".
    metric_type : str
        Metric type, either "error" or "loss".
    y_lim : tuple[float, float] | None, optional
        Common y-axis limits. If None, matplotlib determines the limits
        automatically. The default is None.
    figures_dir : str | Path | None, optional
        Directory where the figure is saved. If None, the figure is not saved.
        The default is None.
    equal_weights : bool, optional
        Whether the figure filename should indicate equal-weight results. The
        default is False.

    Returns
    -------
    None
    """
    column_A0, column_A1 = get_reconstruction_metric_columns(
        method_key=method_key,
        metric_type=metric_type,
    )

    method_name = PREPROCESSING_METHODS[method_key]
    metric_label = (
        "Average reconstruction error"
        if metric_type == "error"
        else "Average reconstruction loss"
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Format Fair PCA reconstruction loss.
    if method_key == "fair_PCA" and metric_type == "loss":
        ax.plot(
            results_df["k"],
            results_df[column_A0],
            marker="o",
            linestyle="-",
            color="C0",
            markevery=slice(0, None, 2),
            label="A = 0",
        )
    
        ax.plot(
            results_df["k"],
            results_df[column_A1],
            marker="s",
            linestyle="--",
            color="C1",
            markevery=slice(1, None, 2),
            label="A = 1",
        )
    
    
    else:
        ax.plot(
            results_df["k"],
            results_df[column_A0],
            marker="o",
            linestyle="-",
            color="C0",
            label="A = 0",
        )
    
        ax.plot(
            results_df["k"],
            results_df[column_A1],
            marker="s",
            linestyle="--",
            color="C1",
            label="A = 1",
        )

    ax.set_title(f"{method_name}: reconstruction {metric_type}")
    ax.set_xlabel("k")
    ax.set_ylabel(metric_label)
    ax.set_xticks(np.arange(2, results_df["k"].max() + 1, 2))
    ax.legend()
    ax.grid(True, alpha=0.3)

    if y_lim is not None:
        ax.set_ylim(y_lim)

    fig.tight_layout()

    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

        figure_name = make_reconstruction_figure_name(
            dataset_name=dataset_name,
            method_key=method_key,
            metric_type=metric_type,
            equal_weights=equal_weights,
        )
        
        output_path = figures_dir / figure_name
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {output_path}")

    plt.show()

def get_common_ylim(
        values: pd.DataFrame | np.ndarray | list,
        metric_type: str | None = None,
        lower_bound_zero: bool = False,
        padding_fraction: float = 0.05
    ) -> tuple[float, float]:
    """
    Compute shared y-axis limits for one or more plots.

    If a reconstruction results dataframe and metric type are provided, the
    limits are computed across all preprocessing methods and both sensitive
    groups for that reconstruction metric. Otherwise, the limits are computed
    directly from the supplied values.

    Parameters
    ----------
    values : pd.DataFrame | np.ndarray | list
        Values used to determine the shared y-axis limits. This can either be a
        reconstruction results dataframe or a general collection of numeric
        values.
    metric_type : str | None, optional
        Reconstruction metric type, either "error" or "loss". If provided,
        values must be a reconstruction results dataframe. The default is None.
    lower_bound_zero : bool, optional
        Whether the lower y-axis limit should be at most zero. The default is
        False.
    padding_fraction : float, optional
        Fraction of the value range added as padding to both limits. The default
        is 0.05.

    Returns
    -------
    tuple[float, float]
        Lower and upper y-axis limits.
    """
    if metric_type is not None:
        if metric_type not in {"error", "loss"}:
            raise ValueError("metric_type must be either 'error' or 'loss'.")

        metric_columns = []

        for method_key in PREPROCESSING_METHODS:
            column_A0, column_A1 = get_reconstruction_metric_columns(
                method_key=method_key,
                metric_type=metric_type,
            )
            metric_columns.extend([column_A0, column_A1])

        values = values[metric_columns].to_numpy().ravel()
    else:
        values = np.asarray(values, dtype=float)

    if values.size == 0:
        raise ValueError("Cannot compute y-axis limits from an empty collection.")

    y_min = float(np.nanmin(values))
    y_max = float(np.nanmax(values))

    if lower_bound_zero:
        y_min = min(0.0, y_min)

    if np.isclose(y_min, y_max):
        padding = 1.0 if np.isclose(y_min, 0.0) else abs(y_min) * padding_fraction
    else:
        padding = (y_max - y_min) * padding_fraction

    return y_min - padding, y_max + padding

def plot_reconstruction_results(
        results_path: str | Path,
        dataset_name: str,
        figures_dir: str | Path | None = None,
        methods: list[str] | None = None,
        equal_weights: bool = False
    ) -> pd.DataFrame:
    """
    Plot reconstruction errors and losses for all selected methods.

    Parameters
    ----------
    results_path : str | Path
        Path to the reconstruction results CSV file.
    dataset_name : str
        Name of the dataset shown in the plot titles.
    figures_dir : str | Path | None, optional
        Directory where figures are saved. If None, figures are not saved. The
        default is None.
    methods : list[str] | None, optional
        Method identifiers to plot. If None, all methods are plotted. The
        default is None.
    equal_weights : bool, optional
        Whether the figure filenames should indicate equal-weight results. The
        default is False.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the loaded reconstruction results.
    """
    results_df = load_reconstruction_results(results_path)

    if methods is None:
        methods = list(PREPROCESSING_METHODS)

    error_ylim = get_common_ylim(
    values=results_df,
    metric_type="error",
    lower_bound_zero=True,
    )
    
    loss_ylim = get_common_ylim(
        values=results_df,
        metric_type="loss",
        lower_bound_zero=False,
    )

    for method_key in methods:
        plot_single_reconstruction_metric(
            results_df=results_df,
            dataset_name=dataset_name,
            method_key=method_key,
            metric_type="error",
            y_lim=error_ylim,
            figures_dir=figures_dir,
            equal_weights=equal_weights,
        )

        plot_single_reconstruction_metric(
            results_df=results_df,
            dataset_name=dataset_name,
            method_key=method_key,
            metric_type="loss",
            y_lim=loss_ylim,
            figures_dir=figures_dir,
            equal_weights=equal_weights,
        )

    return results_df

# =============================================================================
# CROSS-VALIDATION HELPERS
# =============================================================================
def print_CV_results(
        cv_results_df: pd.DataFrame,
        classifier_name: str,
    ) -> None:
    """
    Print a formatted cross-validation summary and LaTeX table rows.

    For each preprocessing method, the function reports:
    - the k with the minimum average validation loss;
    - the k selected by the one-standard-error rule;
    - the corresponding average losses.

    The standard error is used internally to compute the one-standard-error
    threshold, but it is not printed.

    Parameters
    ----------
    cv_results_df : pd.DataFrame
        Dataframe containing cross-validation results. Must contain the columns
        "k", "{method_key}_avg_loss", and "{method_key}_standard_error" for
        each preprocessing method in PREPROCESSING_METHODS.
    classifier_name : str
        Name of the classifier for which the CV results are reported.

    Returns
    -------
    None
        The function prints the formatted summary and LaTeX rows directly.
    """
    results = []

    for method_key, method_name in PREPROCESSING_METHODS.items():
        avg_loss_column = f"{method_key}_avg_loss"
        se_column = f"{method_key}_standard_error"

        min_loss_idx = cv_results_df[avg_loss_column].idxmin()

        min_loss_k = int(cv_results_df.loc[min_loss_idx, "k"])
        min_loss = float(cv_results_df.loc[min_loss_idx, avg_loss_column])
        min_loss_se = float(cv_results_df.loc[min_loss_idx, se_column])

        one_se_threshold = min_loss + min_loss_se

        eligible_rows = cv_results_df[
            cv_results_df[avg_loss_column] <= one_se_threshold
        ]

        one_se_idx = eligible_rows["k"].idxmin()

        one_se_k = int(cv_results_df.loc[one_se_idx, "k"])
        one_se_loss = float(cv_results_df.loc[one_se_idx, avg_loss_column])

        results.append({
            "method_name": method_name,
            "min_loss_k": min_loss_k,
            "min_loss": min_loss,
            "one_se_k": one_se_k,
            "one_se_loss": one_se_loss,
        })
        
    line_width = 92

    print(f"\n{classifier_name} cross-validation k-selection summary")
    print("-" * line_width)
    print(
        f"{'Preprocessing':<18}"
        f"{'Minimum loss k':>22}"
        f"{'Minimum loss':>18}"
        f"{'One-SE k':>14}"
        f"{'One-SE loss':>16}"
    )
    print("-" * line_width)

    for row in results:
        method_name = row["method_name"]
        min_loss_k = row["min_loss_k"]
        min_loss = row["min_loss"]
        one_se_k = row["one_se_k"]
        one_se_loss = row["one_se_loss"]

        print(
            f"{method_name:<18}"
            f"{min_loss_k:>22}"
            f"{min_loss:>18.4f}"
            f"{one_se_k:>14}"
            f"{one_se_loss:>16.4f}"
        )

    print(f"\nLaTeX table rows | {classifier_name}")
    print("-" * line_width)

    for row in results:
        method_name = row["method_name"]
        min_loss_k = row["min_loss_k"]
        min_loss = row["min_loss"]
        one_se_k = row["one_se_k"]
        one_se_loss = row["one_se_loss"]

        print(
            f"{method_name} & "
            f"{min_loss_k} & "
            f"{min_loss:.4f} & "
            f"{one_se_k} & "
            f"{one_se_loss:.4f} \\\\"
        )
        
# =============================================================================
# CLASSIFICATION HELPERS
# =============================================================================
def print_classification_results(
        classification_results_df: pd.DataFrame,
        classifier_name: str,
    ) -> None:
    """
    Print a formatted classification-results summary and LaTeX table rows.

    For each preprocessing method, the function reports overall accuracy,
    protected-group accuracy, demographic disparity, equal opportunity
    difference, and equalized odds difference. All metrics are printed as
    percentages.

    Parameters
    ----------
    classification_results_df : pd.DataFrame
        Dataframe containing true labels, sensitive-variable values, and
        predicted labels for each preprocessing method.
    classifier_name : str
        Name of the classifier for which the classification results are
        reported.

    Returns
    -------
    None
        The function prints the formatted summary and LaTeX rows directly.
    """
    Y = classification_results_df["Y_test"].to_numpy()
    A = classification_results_df["A_test"].to_numpy()

    methods = [
        ("raw_data", "No Preprocessing"),
        ("standard_PCA", "Standard PCA"),
        ("fair_PCA", "Fair PCA"),
        ("LAFTR", "LAFTR"),
    ]

    results = []

    for method_key, method_name in methods:
        Y_pred_column = f"Y_pred_{method_key}"
        Y_pred = classification_results_df[Y_pred_column].to_numpy()

        acc_value = 100 * accuracy(Y, Y_pred)
        acc_prot_value = 100 * accuracy_protected(Y, Y_pred, A)
        disparity_value = 100 * disparity(Y_pred, A)
        opty_value = 100 * opty(Y, Y_pred, A)
        odds_value = 100 * odds(Y, Y_pred, A)

        results.append({
            "method_name": method_name,
            "acc": acc_value,
            "acc_prot": acc_prot_value,
            "disparity": disparity_value,
            "opty": opty_value,
            "odds": odds_value,
        })

    line_width = 82

    print(f"\n{classifier_name} classification results")
    print("-" * line_width)
    print(
        f"{'Preprocessing':<20}"
        f"{'Acc':>10}"
        f"{'AccProt':>10}"
        f"{'Disparity':>12}"
        f"{'Opty':>10}"
        f"{'Odds':>10}"
    )
    print("-" * line_width)

    for row in results:
        method_name = row["method_name"]
        acc_value = row["acc"]
        acc_prot_value = row["acc_prot"]
        disparity_value = row["disparity"]
        opty_value = row["opty"]
        odds_value = row["odds"]

        print(
            f"{method_name:<20}"
            f"{acc_value:>10.2f}"
            f"{acc_prot_value:>10.2f}"
            f"{disparity_value:>12.2f}"
            f"{opty_value:>10.2f}"
            f"{odds_value:>10.2f}"
        )

    print(f"\nLaTeX table rows | {classifier_name}")
    print("-" * line_width)

    for row in results:
        method_name = row["method_name"]
        acc_value = row["acc"]
        acc_prot_value = row["acc_prot"]
        disparity_value = row["disparity"]
        opty_value = row["opty"]
        odds_value = row["odds"]

        print(
            f"{method_name} & "
            f"{acc_value:.2f} & "
            f"{acc_prot_value:.2f} & "
            f"{disparity_value:.2f} & "
            f"{opty_value:.2f} & "
            f"{odds_value:.2f} \\\\"
        )
