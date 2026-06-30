"""
Data loading and preparing utilities for the Adult and COMPAS datasets.

This module provides dataset-specific loading functions and wrapper functions
that return NumPy arrays for use in downstream experiments. The loading
functions apply the required data preparation steps, including filtering,
discretization, one-hot encoding, label construction, and sensitive-variable
selection. The wrapper functions convert the processed dataframes into feature
matrices, sensitive-variable vectors, and label vectors.

The sensitive variable is returned separately from the feature matrix. For
Adult, gender is used as the sensitive variable, with female encoded as 1. For
COMPAS, race is used as the sensitive variable, with African-American encoded
as 1 and Caucasian encoded as 0.
"""

import numpy as np
import pandas as pd
import os
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
COMPAS_DIR = DATA_DIR / "compas"
ADULT_DIR = DATA_DIR / "adult"

def get_compas_data(
        data_folder: str,
        sensitive: str = "race",
        flip: bool = True
    ) -> tuple[pd.DataFrame, list[str], str, str]:
    """
    Load and preprocess the COMPAS dataset.

    The COMPAS dataset is loaded from ``compas-scores-two-years.csv``. The data
    is filtered using the standard COMPAS selection criteria, selected
    continuous variables are discretized, and categorical variables are one-hot
    encoded. The sensitive variable is returned separately and removed from the
    explanatory feature set.

    Parameters
    ----------
    data_folder : str
        Directory containing ``compas-scores-two-years.csv``.
    sensitive : str, optional
        Sensitive variable used to define protected-group membership. Currently,
        only ``"race"`` is supported. The default is ``"race"``.
    flip : bool, optional
        Whether to flip ``two_year_recid`` into ``two_year_norecid``, so that
        1 indicates no recidivism. The default is True.

    Returns
    -------
    df : pandas.DataFrame
        Preprocessed COMPAS dataframe containing the selected feature columns,
        the sensitive-variable column, and the label column.
    features : list[str]
        Names of the feature columns used as explanatory variables.
    sensitive_column : str
        Name of the sensitive-variable indicator column.
    label : str
        Name of the binary label column.
    """
    if sensitive != "race":
        raise ValueError("Currently, only sensitive='race' is supported for COMPAS.")

    path_to_compas_data = os.path.join(data_folder, "compas-scores-two-years.csv")
    df = pd.read_csv(path_to_compas_data)

    selected_columns = [
        "age", "c_charge_degree", "race", "age_cat", "score_text", "sex",
        "priors_count", "days_b_screening_arrest", "decile_score", "is_recid",
        "two_year_recid"
    ]

    df = df[selected_columns].copy()

    # Apply standard COMPAS filtering criteria.
    df = df[df["days_b_screening_arrest"] <= 30]
    df = df[df["days_b_screening_arrest"] >= -30]
    df = df[df["is_recid"] != -1]
    df = df[df["c_charge_degree"] != "O"]
    df = df[df["score_text"] != "N/A"]

    # Restrict the race comparison to African-American and Caucasian defendants.
    df = df[df["race"].isin(["African-American", "Caucasian"])].copy()

    continuous_to_categorical_features = ["age", "decile_score", "priors_count"]
    categorical_features = ["c_charge_degree", "race", "score_text", "sex"]

    # =========================================================================
    # Utility functions
    # -------------------------------------------------------------------------
    def binarize_categorical_columns(
            input_df: pd.DataFrame,
            categorical_columns: list[str]
        ) -> pd.DataFrame:
        """
        One-hot encode categorical columns.
        """
        return pd.get_dummies(input_df, columns=categorical_columns)

    def bucketize_continuous_column(
            input_df: pd.DataFrame,
            continuous_column_name: str,
            bins: list[float]
        ) -> None:
        """
        Bucketize a continuous column using fixed bin edges.
        """
        input_df[continuous_column_name] = pd.cut(
            input_df[continuous_column_name],
            bins,
            labels=False
        )
    # =========================================================================

    for column in continuous_to_categorical_features:
        bins = [0] + list(np.percentile(df[column], [20, 40, 60, 80, 90, 100]))

        if column == "priors_count":
            bins = list(np.percentile(df[column], [0, 50, 70, 80, 90, 100]))

        bucketize_continuous_column(df, column, bins=bins)

    df = binarize_categorical_columns(
        df,
        categorical_columns=categorical_features + continuous_to_categorical_features
    )

    # Create cumulative threshold dummy columns for decile score.
    decile_columns = [
        "decile_score_0", "decile_score_1", "decile_score_2",
        "decile_score_3", "decile_score_4", "decile_score_5"
    ]

    for i in range(len(decile_columns) - 1):
        df[decile_columns[i]] = df[decile_columns[i:]].max(axis=1)

    # Create cumulative threshold dummy columns for prior counts.
    priors_columns = [
        "priors_count_0.0", "priors_count_1.0", "priors_count_2.0",
        "priors_count_3.0", "priors_count_4.0"
    ]

    for i in range(len(priors_columns) - 1):
        df[priors_columns[i]] = df[priors_columns[i:]].max(axis=1)

    label = "two_year_recid"
    sensitive_column = "race_African-American"

    features = [
        "days_b_screening_arrest",
        "c_charge_degree_F",
        "c_charge_degree_M",
        "score_text_High",
        "score_text_Low",
        "score_text_Medium",
        "sex_Female",
        "sex_Male",
        "age_0",
        "age_1",
        "age_2",
        "age_3",
        "age_4",
        "age_5",
        "decile_score_0",
        "decile_score_1",
        "decile_score_2",
        "decile_score_3",
        "decile_score_4",
        "decile_score_5",
        "priors_count_0.0",
        "priors_count_1.0",
        "priors_count_2.0",
        "priors_count_3.0",
        "priors_count_4.0"
    ]
    
    df = df[features + [sensitive_column] + [label]]

    if flip:
        removed_label = df.pop(label).to_numpy()
        label = "two_year_norecid"
        df[label] = 1.0 - removed_label

    return df, features, sensitive_column, label

def get_adult_data(
        data_folder: str,
        sensitive: str = "gender",
        shuffle_seed: int = 0
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str, str]:
    """
    Load and preprocess the Adult dataset.
    
    The Adult dataset is loaded from its standard train-test files:
    ``adult.data`` is used as the training file and ``adult.test`` is used as the
    testing file. The target variable is converted to a binary label, selected
    continuous variables are discretized, and categorical variables are one-hot
    encoded. The sensitive variable is returned separately and removed from the
    explanatory feature set. The resulting train and test dataframes are aligned 
    to ensure that both contain the same dummy-variable columns.
    
    Parameters
    ----------
    data_folder : str
        Directory containing ``adult.data`` and ``adult.test``.
    sensitive : str, optional
        Sensitive variable used to define protected-group membership. Currently,
        only ``"gender"`` is supported. The default is ``"gender"``.
    shuffle_seed : int, optional
        Random seed used to shuffle the processed train and test dataframes. The
        default is 0.
    
    Returns
    -------
    train_df : pandas.DataFrame
        Preprocessed Adult training dataframe.
    test_df : pandas.DataFrame
        Preprocessed Adult testing dataframe.
    features : list[str]
        Names of the feature columns used as explanatory variables.
    sensitive_column : str
        Name of the sensitive-variable indicator column.
    label : str
        Name of the binary label column.
    """
    
    if sensitive != "gender":
        raise ValueError("Currently, only sensitive='gender' is supported for Adult.")
    
    path_to_adult_train = os.path.join(data_folder, "adult.data")
    path_to_adult_test = os.path.join(data_folder, "adult.test")
    
    categorical_columns = [
        "workclass", "education", "marital_status", "occupation",
        "relationship", "race", "gender", "native_country"
    ]

    continuous_columns = [
        "age", "capital_gain", "capital_loss", "hours_per_week",
        "education_num"
    ]

    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "gender",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income_bracket"
    ]
    
    label = "label"

    train_df_raw = pd.read_csv(
        path_to_adult_train,
        names=columns,
        skipinitialspace=True
    )

    test_df_raw = pd.read_csv(
        path_to_adult_test,
        names=columns,
        skipinitialspace=True,
        skiprows=1
    )
    
    # CREATE BINARY LABEL
    train_df_raw[label] = train_df_raw["income_bracket"].apply(
        lambda x: ">50K" in str(x)
    ).astype(int)
    
    test_df_raw[label] = test_df_raw["income_bracket"].apply(
        lambda x: ">50K" in str(x)
    ).astype(int)
    
    train_df = train_df_raw[categorical_columns + continuous_columns + [label]].copy()
    test_df = test_df_raw[categorical_columns + continuous_columns + [label]].copy()
    
    # =========================================================================
    # UTILITY FUNCTIONS
    # -------------------------------------------------------------------------
    def binarize_categorical_columns(
            input_train_df: pd.DataFrame,
            input_test_df: pd.DataFrame,
            categorical_columns: list[str]
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        One-hot encode categorical columns and align train/test columns.
        """
        binarized_train_df = pd.get_dummies(
            input_train_df,
            columns=categorical_columns
        )
    
        binarized_test_df = pd.get_dummies(
            input_test_df,
            columns=categorical_columns
        )
    
        test_missing_cols = set(binarized_train_df.columns) - set(binarized_test_df.columns)
        for c in test_missing_cols:
            binarized_test_df[c] = 0
    
        train_missing_cols = set(binarized_test_df.columns) - set(binarized_train_df.columns)
        for c in train_missing_cols:
            binarized_train_df[c] = 0
    
        binarized_test_df = binarized_test_df[binarized_train_df.columns]
    
        return binarized_train_df, binarized_test_df
     
    def bucketize_continuous_column(
            input_train_df: pd.DataFrame,
            input_test_df: pd.DataFrame,
            continuous_column_name: str,
            num_quantiles: int | None = None,
            bins: list[float] | None = None
        ) -> None:
        """
        Bucketize a continuous column in train and test.

        Quantile bins are estimated on the training set only and then applied to
        both train and test.
        """
        if num_quantiles is not None and bins is not None:
            raise ValueError("Specify either num_quantiles or bins, not both.")

        if num_quantiles is not None:
            _, bins_quantized = pd.qcut(
                input_train_df[continuous_column_name],
                num_quantiles,
                retbins=True,
                labels=False,
                duplicates="drop"
            )

            input_train_df[continuous_column_name] = pd.cut(
                input_train_df[continuous_column_name],
                bins_quantized,
                labels=False,
                include_lowest=True
            )

            input_test_df[continuous_column_name] = pd.cut(
                input_test_df[continuous_column_name],
                bins_quantized,
                labels=False,
                include_lowest=True
            )

        elif bins is not None:
            input_train_df[continuous_column_name] = pd.cut(
                input_train_df[continuous_column_name],
                bins,
                labels=False,
                include_lowest=True
            )

            input_test_df[continuous_column_name] = pd.cut(
                input_test_df[continuous_column_name],
                bins,
                labels=False,
                include_lowest=True
            )

        else:
            raise ValueError("Either num_quantiles or bins must be specified.")
    # =========================================================================
    
    bucketize_continuous_column(train_df, test_df, "age", num_quantiles=4)

    bucketize_continuous_column(
        train_df, test_df, "capital_gain",
        bins=[-1, 1, 4000, 10000, 100000]
    )

    bucketize_continuous_column(
        train_df, test_df, "capital_loss",
        bins=[-1, 1, 1800, 1950, 4500]
    )

    bucketize_continuous_column(
        train_df, test_df, "hours_per_week",
        bins=[0, 39, 41, 50, 100]
    )

    bucketize_continuous_column(
        train_df, test_df, "education_num",
        bins=[0, 8, 9, 11, 16]
    )

    train_df, test_df = binarize_categorical_columns(
        train_df,
        test_df,
        categorical_columns=categorical_columns + continuous_columns
    )

    sensitive_column = "gender_Female"
    other_sensitive_column = "gender_Male"

    features = list(train_df.columns)
    features.remove(label)
    features.remove(sensitive_column)
    features.remove(other_sensitive_column)

    train_df = train_df[features + [sensitive_column] + [label]]
    test_df = test_df[features + [sensitive_column] + [label]]

    train_df = train_df.sample(
        frac=1.0,
        random_state=shuffle_seed
    ).reset_index(drop=True)

    test_df = test_df.sample(
        frac=1.0,
        random_state=shuffle_seed
    ).reset_index(drop=True)

    return train_df, test_df, features, sensitive_column, label
    
def wrap_COMPAS_full(
        base_folder: str | Path | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the full preprocessed COMPAS dataset.

    Parameters
    ----------
    base_folder : str, pathlib.Path, or None, optional
        Directory containing the COMPAS data file. If None, ``COMPAS_DIR`` is used.

    Returns
    -------
    X_COMPAS : numpy.ndarray
        Feature matrix.
    A_COMPAS : numpy.ndarray
        Sensitive-variable vector.
    Y_COMPAS : numpy.ndarray
        Label vector.
    """
    if base_folder is None:
        base_folder = COMPAS_DIR

    df, features, sensitive_column, label = get_compas_data(str(base_folder))

    X_COMPAS = df[features].to_numpy(dtype=float)
    A_COMPAS = df[sensitive_column].to_numpy(dtype=int)
    Y_COMPAS = df[label].to_numpy(dtype=int)

    return X_COMPAS, A_COMPAS, Y_COMPAS

def wrap_COMPAS_split(
        base_folder: str | Path | None = None,
        train_size: float = 0.8,
        seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the preprocessed COMPAS dataset and create a train-test split.

    Parameters
    ----------
    base_folder : str, pathlib.Path, or None, optional
        Directory containing the COMPAS data file. If None, ``COMPAS_DIR`` is used.
    train_size : float, optional
        Share of observations used for training. The default is 0.8.
    seed : int, optional
        Random seed for the split. The default is 42.

    Returns
    -------
    X_train, A_train, Y_train : numpy.ndarray
        Training features, sensitive variable, and labels.
    X_test, A_test, Y_test : numpy.ndarray
        Testing features, sensitive variable, and labels.
    """
    if not 0.0 < train_size < 1.0:
        raise ValueError("train_size must be strictly between 0 and 1.")

    set_seed(seed)

    X_COMPAS, A_COMPAS, Y_COMPAS = wrap_COMPAS_full(base_folder)
    n_samples = X_COMPAS.shape[0]
    train_count = int(train_size * n_samples)

    indices = np.random.permutation(n_samples)
    train_indices = indices[:train_count]
    test_indices = indices[train_count:]

    X_train = X_COMPAS[train_indices]
    A_train = A_COMPAS[train_indices]
    Y_train = Y_COMPAS[train_indices]

    X_test = X_COMPAS[test_indices]
    A_test = A_COMPAS[test_indices]
    Y_test = Y_COMPAS[test_indices]

    return X_train, A_train, Y_train, X_test, A_test, Y_test

def wrap_COMPAS_equal_weights(
        base_folder: str | Path | None = None,
        seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the full preprocessed COMPAS dataset with an equal number of protected
    and unprotected observations.

    Parameters
    ----------
    base_folder : str, pathlib.Path, or None, optional
        Directory containing the COMPAS data file. If None, ``COMPAS_DIR`` is used.
    seed : int, optional
        Random seed used for group sampling and final shuffling. The default is 42.

    Returns
    -------
    X_equal : numpy.ndarray
        Feature matrix with balanced protected and unprotected groups.
    A_equal : numpy.ndarray
        Sensitive-variable vector with equal numbers of 0 and 1 values.
    Y_equal : numpy.ndarray
        Label vector corresponding to the balanced feature matrix.
    """
    X_COMPAS, A_COMPAS, Y_COMPAS = wrap_COMPAS_full(base_folder)

    rng = np.random.default_rng(seed)

    protected_indices = np.where(A_COMPAS == 1)[0]
    unprotected_indices = np.where(A_COMPAS == 0)[0]

    n_equal = min(len(protected_indices), len(unprotected_indices))

    sampled_protected_indices = rng.choice(
        protected_indices,
        size=n_equal,
        replace=False
    )

    sampled_unprotected_indices = rng.choice(
        unprotected_indices,
        size=n_equal,
        replace=False
    )

    balanced_indices = np.concatenate([
        sampled_unprotected_indices,
        sampled_protected_indices
    ])

    shuffled_indices = rng.permutation(balanced_indices)

    X_equal = X_COMPAS[shuffled_indices]
    A_equal = A_COMPAS[shuffled_indices]
    Y_equal = Y_COMPAS[shuffled_indices]

    return X_equal, A_equal, Y_equal

def wrap_Adult_full(
        base_folder: str | Path | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the full preprocessed Adult dataset.

    Parameters
    ----------
    base_folder : str, pathlib.Path, or None, optional
        Directory containing the Adult data files. If None, ``ADULT_DIR`` is used.

    Returns
    -------
    X_adult : numpy.ndarray
        Feature matrix.
    A_adult : numpy.ndarray
        Sensitive-variable vector.
    Y_adult : numpy.ndarray
        Label vector.
    """
    if base_folder is None:
        base_folder = ADULT_DIR

    train_df, test_df, features, sensitive_column, label = get_adult_data(
        data_folder=str(base_folder)
    )

    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    X_adult = df[features].to_numpy(dtype=float)
    A_adult = df[sensitive_column].to_numpy(dtype=int)
    Y_adult = df[label].to_numpy(dtype=int)

    return X_adult, A_adult, Y_adult

def wrap_Adult_split(
        base_folder: str | Path | None = None,
        use_standard_split: bool = True,
        train_size: float = 0.8,
        seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the preprocessed Adult dataset and create a train-test split.

    Parameters
    ----------
    base_folder : str, pathlib.Path, or None, optional
        Directory containing the Adult data files. If None, ``ADULT_DIR`` is used.
    use_standard_split : bool, optional
        Whether to use the original Adult train-test split. The default is True.
    train_size : float, optional
        Share of observations used for training if ``use_standard_split`` is False.
        The default is 0.8.
    seed : int, optional
        Random seed for the random split. The default is 42.

    Returns
    -------
    X_train, A_train, Y_train : numpy.ndarray
        Training features, sensitive variable, and labels.
    X_test, A_test, Y_test : numpy.ndarray
        Testing features, sensitive variable, and labels.
    """
    if not 0.0 < train_size < 1.0:
        raise ValueError("train_size must be strictly between 0 and 1.")

    if base_folder is None:
        base_folder = ADULT_DIR

    train_df, test_df, features, sensitive_column, label = get_adult_data(
        data_folder=str(base_folder)
    )

    if not use_standard_split:
        set_seed(seed)

        df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

        train_count = int(train_size * df.shape[0])
        indices = np.random.permutation(df.shape[0])

        train_indices = indices[:train_count]
        test_indices = indices[train_count:]

        train_df = df.iloc[train_indices].reset_index(drop=True)
        test_df = df.iloc[test_indices].reset_index(drop=True)

    X_train = train_df[features].to_numpy(dtype=float)
    A_train = train_df[sensitive_column].to_numpy(dtype=int)
    Y_train = train_df[label].to_numpy(dtype=int)

    X_test = test_df[features].to_numpy(dtype=float)
    A_test = test_df[sensitive_column].to_numpy(dtype=int)
    Y_test = test_df[label].to_numpy(dtype=int)

    return X_train, A_train, Y_train, X_test, A_test, Y_test

def wrap_Adult_equal_weights(
        base_folder: str | Path | None = None,
        seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the full preprocessed Adult dataset with an equal number of protected
    and unprotected observations.

    Parameters
    ----------
    base_folder : str, pathlib.Path, or None, optional
        Directory containing the Adult data files. If None, ``ADULT_DIR`` is used.
    seed : int, optional
        Random seed used for group sampling and final shuffling. The default is 42.

    Returns
    -------
    X_equal : numpy.ndarray
        Feature matrix with balanced protected and unprotected groups.
    A_equal : numpy.ndarray
        Sensitive-variable vector with equal numbers of 0 and 1 values.
    Y_equal : numpy.ndarray
        Label vector corresponding to the balanced feature matrix.
    """
    X_Adult, A_Adult, Y_Adult = wrap_Adult_full(base_folder)

    rng = np.random.default_rng(seed)

    protected_indices = np.where(A_Adult == 1)[0]
    unprotected_indices = np.where(A_Adult == 0)[0]

    n_equal = min(len(protected_indices), len(unprotected_indices))

    sampled_protected_indices = rng.choice(
        protected_indices,
        size=n_equal,
        replace=False
    )

    sampled_unprotected_indices = rng.choice(
        unprotected_indices,
        size=n_equal,
        replace=False
    )

    balanced_indices = np.concatenate([
        sampled_unprotected_indices,
        sampled_protected_indices
    ])

    shuffled_indices = rng.permutation(balanced_indices)

    X_equal = X_Adult[shuffled_indices]
    A_equal = A_Adult[shuffled_indices]
    Y_equal = Y_Adult[shuffled_indices]

    return X_equal, A_equal, Y_equal

def set_seed(seed: int = 42) -> None:
    """
    Set seeds for Python system-level and NumPy random generation.

    Parameters
    ----------
    seed : int, optional
        Random seed used for reproducible results. The default is 42.

    Returns
    -------
    None.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)