"""
Fair PCA implementation.

This module defines a FairPCA class that solves a variant of the Fair PCA 
algorithm of Samadi et al. (2018), and provides methods for dimensionality 
reduction and reconstruction.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import cvxpy as cp
from scipy.optimize import linprog

from preprocessing.standard_pca import StandardPCA
from preprocessing.utils import frobenius_squared

@dataclass
class FairPCA:
    """
    Fair PCA preprocessing model.
    
    This class implements the Fair PCA algorithm used in the thesis, following
    the reconstruction-fairness approach of Samadi et al. (2018). It fits a
    group-aware loading matrix for dimensionality reduction and reconstruction.
    """
    # USER INITIALISATION
    k: int
    center: bool = True
    sdp_solver: str = "SCS"
    sdp_eps: float = 1e-6
    verbose: bool = False

    # STANDARD INITIALISATION
    V: np.ndarray = field(init=False)
    mean_protected: np.ndarray = field(init=False)
    mean_unprotected: np.ndarray = field(init=False)

    P_hat: np.ndarray = field(init=False)
    P_star: np.ndarray = field(init=False)

    lambda_bar: np.ndarray = field(init=False)
    lambda_star: np.ndarray = field(init=False)
    eigenvalues: np.ndarray = field(init=False)

    rank_fair: int = field(init=False)
    rank_U: int = field(init=False)

    z_sdp: float = field(init=False)
    z_lp: float = field(init=False)

    fitted: bool = field(default=False, init=False)
    
    def fit(self, X: np.ndarray, A: np.ndarray) -> FairPCA:
        """
        Fit the Fair PCA model.
    
        The method applies the Fair PCA solving algorithm. For more details on
        the solving algorithm, see Algorithm 1 in the accompanying thesis paper
        or see the original work of Samadi et al. (2018).
    
        Parameters
        ----------
        X : np.ndarray
            Input feature matrix of shape (n_samples, n_features).
        A : np.ndarray
            Sensitive variable vector of shape (n_samples,), encoded as 0 and 1.
    
        Returns
        -------
        FairPCA
            The fitted Fair PCA instance.
        """
        X_0, X_1 = self._split_sample(X, A)
    
        if X_0.shape[0] == 0:
            raise ValueError("X_0 must contain at least one row.")
    
        if X_1.shape[0] == 0:
            raise ValueError("X_1 must contain at least one row.")
    
        m0, n = X_0.shape
        m1, _ = X_1.shape
    
        if not isinstance(self.k, int):
            raise TypeError("k must be an integer.")
    
        if self.k <= 0 or self.k >= n:
            raise ValueError("k must satisfy 0 < k < number of features.")
    
        if self.center:
            mean_0 = X_0.mean(axis=0)
            mean_1 = X_1.mean(axis=0)
    
            X_0 = X_0 - mean_0
            X_1 = X_1 - mean_1
        else:
            mean_0 = np.zeros(n)
            mean_1 = np.zeros(n)
    
        # ---------------------------------------------------------------------
        # Step 1: Group-specific optimal rank-k approximations
        # ---------------------------------------------------------------------
        standard_pca = StandardPCA(k=self.k, center=False)
    
        X_0_hat = standard_pca.fit_reconstruct(X_0)
        X_1_hat = standard_pca.fit_reconstruct(X_1)
    
        # ---------------------------------------------------------------------
        # Step 2: Solve semidefinite optimization problem
        # ---------------------------------------------------------------------
        X_0_hat_norm_sq = frobenius_squared(X_0_hat)
        X_1_hat_norm_sq = frobenius_squared(X_1_hat)
    
        X0tX0 = X_0.T @ X_0
        X1tX1 = X_1.T @ X_1
        
        P = cp.Variable((n, n), symmetric=True)
        z = cp.Variable()
        I = np.eye(n)
    
        constraints = [
            z >= (X_0_hat_norm_sq - cp.trace(X0tX0 @ P)) / m0,
            z >= (X_1_hat_norm_sq - cp.trace(X1tX1 @ P)) / m1,
            cp.trace(P) <= self.k,
            P >> 0,
            I - P >> 0,
        ]
    
        problem = cp.Problem(cp.Minimize(z), constraints)
    
        solve_kwargs = {"verbose": self.verbose}
        if self.sdp_solver.upper() == "SCS":
            solve_kwargs.update({"eps": self.sdp_eps, "max_iters": 100_000})
    
        problem.solve(solver=self.sdp_solver, **solve_kwargs)
    
        if problem.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"SDP did not solve successfully. Status: {problem.status}")
    
        # Ensure symmetry, since numerical solvers may return tiny asymmetries.
        P_hat = np.asarray(P.value, dtype=float)
        P_hat = 0.5 * (P_hat + P_hat.T)
    
        # ---------------------------------------------------------------------
        # Step 3: Apply eigenvalue decomposition to P_hat
        # ---------------------------------------------------------------------
        eigenvalues, eigenvectors = np.linalg.eigh(P_hat)
    
        # Sort eigenvalues in descending order.
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
    
        # Numerical cleanup of possible infeasibilities caused by the solver.
        eigenvalues = np.clip(eigenvalues, 0.0, 1.0)
    
        # ---------------------------------------------------------------------
        # Step 4: Solve LP over lambda_bar and z
        #
        # Variables are x = [lambda_1, ..., lambda_n, z]
        # ---------------------------------------------------------------------
        # Compute how much variance each group has in direction u_j.
        coeff_0 = np.array([
            eigenvectors[:, j].T @ X0tX0 @ eigenvectors[:, j]
            for j in range(n)
        ])
    
        coeff_1 = np.array([
            eigenvectors[:, j].T @ X1tX1 @ eigenvectors[:, j]
            for j in range(n)
        ])
        
        c = np.zeros(n + 1)
        c[-1] = 1.0
    
        A_ub = []
        b_ub = []
    
        # Constraint 1:
        # z >= (X_0_hat_norm_sq - coeff_0 @ lambda) / m0
        # <=> -coeff_0 @ lambda - m0 * z <= -X_0_hat_norm_sq
        row_0 = np.zeros(n + 1)
        row_0[:n] = -coeff_0
        row_0[-1] = -m0
        A_ub.append(row_0)
        b_ub.append(-X_0_hat_norm_sq)
    
        # Constraint 2:
        # z >= (X_1_hat_norm_sq - coeff_1 @ lambda) / m1
        # <=> -coeff_1 @ lambda - m1 * z <= -X_1_hat_norm_sq
        row_1 = np.zeros(n + 1)
        row_1[:n] = -coeff_1
        row_1[-1] = -m1
        A_ub.append(row_1)
        b_ub.append(-X_1_hat_norm_sq)
    
        # Constraint 3:
        # sum lambda_i <= k
        row_sum = np.zeros(n + 1)
        row_sum[:n] = 1.0
        A_ub.append(row_sum)
        b_ub.append(self.k)
    
        bounds = [(0.0, 1.0)] * n + [(0.0, None)]
    
        lp_result = linprog(
            c=c,
            A_ub=np.asarray(A_ub),
            b_ub=np.asarray(b_ub),
            bounds=bounds,
            method="highs-ds",
        )
    
        if not lp_result.success:
            raise RuntimeError(f"LP did not solve successfully: {lp_result.message}")
    
        lambda_bar = lp_result.x[:n]
        z_lp = float(lp_result.x[-1])
    
        # Numerical cleanup of possible infeasibilities caused by the solver.
        lambda_bar = np.clip(lambda_bar, 0.0, 1.0)
    
        # ---------------------------------------------------------------------
        # Step 5: Compute lambda_star and P_star
        # ---------------------------------------------------------------------
        lambda_star = 1.0 - np.sqrt(np.maximum(0.0, 1.0 - lambda_bar))
    
        # P_star is solution to original algorithm of Samadi et al. (2018).
        P_star = eigenvectors @ np.diag(lambda_star) @ eigenvectors.T
        P_star = 0.5 * (P_star + P_star.T)
    
        # ---------------------------------------------------------------------
        # Step 6: Define set of active eigenvalues
        # ---------------------------------------------------------------------
        tol = 1e-8
        active = lambda_star > tol
    
        # ---------------------------------------------------------------------
        # Step 7: Construct fair PCA loading matrix V
        # ---------------------------------------------------------------------    
        eigenvectors_active = eigenvectors[:, active]
        lambda_star_active = lambda_star[active]
        V = eigenvectors_active @ np.diag(np.sqrt(lambda_star_active))
        
        # ---------------------------------------------------------------------
        # Step 8: Compute reduced representations and check validity
        # ---------------------------------------------------------------------
        X_centered = np.vstack([X_0, X_1])
        Z = X_centered @ V
        X_hat = Z @ V.T
    
        # Check whether our solution corresponds to solution of original algorithm.
        if not np.allclose(P_star, V @ V.T, atol=1e-6):
            raise RuntimeError("Numerical check failed: P_star is not approximately V @ V.T.")
    
        if not np.allclose(X_hat, X_centered @ P_star, atol=1e-6):
            raise RuntimeError("Numerical check failed: X_hat is not approximately X_centered @ P_star.")
    
        # -------------------------------------------------------------------------
        # Store fitted Fair PCA attributes
        # -------------------------------------------------------------------------
        self.eigenvalues = eigenvalues[:self.k]
        self.V = V
    
        self.mean_unprotected = mean_0
        self.mean_protected = mean_1
    
        self.P_hat = P_hat
        self.P_star = P_star
    
        self.lambda_bar = lambda_bar
        self.lambda_star = lambda_star
    
        self.rank_fair = int(np.sum(active))
        self.rank_U = int(np.linalg.matrix_rank(X_hat, tol=tol))
    
        self.z_sdp = float(problem.value)
        self.z_lp = z_lp
    
        self.fitted = True
    
        return self

    def transform(self, X: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform data to the reduced Fair PCA representation.
    
        Parameters
        ----------
        X : np.ndarray
            Input feature matrix of shape (n_samples, n_features).
        A : np.ndarray
            Sensitive variable vector of shape (n_samples,), encoded as 0 and 1.
    
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Reduced representations Z_0 and Z_1, where Z_0 contains the observations
            with A = 0 and Z_1 contains the observations with A = 1.
        """
        self._check_is_fitted()
    
        X_0, X_1 = self._split_sample(X, A)
    
        if X_0.shape[1] != self.V.shape[0]:
            raise ValueError("X must have the same number of features as the fitted data.")
    
        if X_1.shape[1] != self.V.shape[0]:
            raise ValueError("X must have the same number of features as the fitted data.")
    
        X_0_centered = X_0 - self.mean_unprotected
        X_1_centered = X_1 - self.mean_protected
    
        Z_0 = X_0_centered @ self.V
        Z_1 = X_1_centered @ self.V
    
        return Z_0, Z_1

    def reconstruct_full(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from its reduced Fair PCA representation.
        
        Parameters
        ----------
        X : np.ndarray
            Input feature matrix of shape (n_samples, n_features).
        A : np.ndarray
            Sensitive variable vector of shape (n_samples,), encoded as 0 and 1.
        
        Returns
        -------
        np.ndarray
            Fair PCA reconstruction of X, returned in the original row order.
        """
        self._check_is_fitted()
        
        X = self._validate_features(X)
        A = self._validate_sensitives(A)
        
        if X.shape[0] != A.shape[0]:
            raise ValueError("X and A must contain the same number of observations.")
        
        if X.shape[1] != self.V.shape[0]:
            raise ValueError("X must have the same number of features as the fitted data.")
        
        X_hat = np.empty_like(X, dtype=float)
        
        mask_0 = A == 0
        mask_1 = A == 1
        
        X_0 = X[mask_0]
        X_1 = X[mask_1]
        
        X_0_centered = X_0 - self.mean_unprotected
        X_1_centered = X_1 - self.mean_protected
        
        Z_0 = X_0_centered @ self.V
        Z_1 = X_1_centered @ self.V
        
        X_0_hat_centered = Z_0 @ self.V.T
        X_1_hat_centered = Z_1 @ self.V.T
        
        if self.center:
            X_hat[mask_0] = X_0_hat_centered + self.mean_unprotected
            X_hat[mask_1] = X_1_hat_centered + self.mean_protected
        else:
            X_hat[mask_0] = X_0_hat_centered
            X_hat[mask_1] = X_1_hat_centered
        
        return X_hat

    def reconstruct_split(self, X: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct data from its reduced Fair PCA representation and return it split by group.
    
        Parameters
        ----------
        X : np.ndarray
            Input feature matrix of shape (n_samples, n_features).
        A : np.ndarray
            Sensitive variable vector of shape (n_samples,), encoded as 0 and 1.
    
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Reconstructed feature matrices X_0_hat and X_1_hat, where X_0_hat
            contains the reconstructed observations with A = 0 and X_1_hat contains
            the reconstructed observations with A = 1.
        """
        X_hat = self.reconstruct_full(X, A)
        A = self._validate_sensitives(A)
    
        return X_hat[A == 0], X_hat[A == 1]

    def fit_transform(self, X: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit the Fair PCA model and transform the data to the reduced representation.
    
        Parameters
        ----------
        X : np.ndarray
            Input feature matrix of shape (n_samples, n_features).
        A : np.ndarray
            Sensitive variable vector of shape (n_samples,), encoded as 0 and 1.
    
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Reduced representations Z_0 and Z_1, where Z_0 contains the observations
            with A = 0 and Z_1 contains the observations with A = 1.
        """
        self.fit(X, A)
        return self.transform(X, A)
    
    
    def fit_reconstruct_full(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        Fit the Fair PCA model and reconstruct the full input matrix.
    
        Parameters
        ----------
        X : np.ndarray
            Input feature matrix of shape (n_samples, n_features).
        A : np.ndarray
            Sensitive variable vector of shape (n_samples,), encoded as 0 and 1.
    
        Returns
        -------
        np.ndarray
            Fair PCA reconstruction of X, returned in the original row order.
        """
        self.fit(X, A)
        return self.reconstruct_full(X, A)
    
    
    def fit_reconstruct_split(self, X: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit the Fair PCA model and reconstruct the input matrix split by group.
    
        Parameters
        ----------
        X : np.ndarray
            Input feature matrix of shape (n_samples, n_features).
        A : np.ndarray
            Sensitive variable vector of shape (n_samples,), encoded as 0 and 1.
    
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Reconstructed feature matrices X_0_hat and X_1_hat, where X_0_hat
            contains the reconstructed observations with A = 0 and X_1_hat contains
            the reconstructed observations with A = 1.
        """
        self.fit(X, A)
        return self.reconstruct_split(X, A)
    
    
    def get_k(self) -> int:
        """
        Return the requested Fair PCA target dimension.
    
        Returns
        -------
        int
            Target dimension k.
        """
        return self.k
    
    
    def get_rank_fair(self) -> int:
        """
        Return the effective rank of the fitted Fair PCA representation.
    
        Returns
        -------
        int
            Effective Fair PCA rank.
        """
        self._check_is_fitted()
        return self.rank_fair    
    
    def get_loadings(self) -> np.ndarray:
        """
        Return the fitted Fair PCA loading matrix.
    
        Returns
        -------
        np.ndarray
            Loading matrix of shape (n_features, rank_fair).
        """
        self._check_is_fitted()
        return self.V
    
    @staticmethod
    def _validate_features(X: np.ndarray) -> np.ndarray:
        """
        Validate and convert a feature matrix.
    
        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.
    
        Returns
        -------
        np.ndarray
            Feature matrix converted to float.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy ndarray.")
    
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional matrix.")
    
        if not np.issubdtype(X.dtype, np.number):
            raise TypeError("X must contain numeric values.")
    
        return X.astype(float)
    
    @staticmethod
    def _validate_sensitives(A: np.ndarray) -> np.ndarray:
        """
        Validate and convert a sensitive variable vector.

        Parameters
        ----------
        A : np.ndarray
            Input sensitive variable vector.

        Returns
        -------
        np.ndarray
            Sensitive variable vector converted to integer values.
        """
        if not isinstance(A, np.ndarray):
            raise TypeError("A must be a NumPy ndarray.")

        if A.ndim != 1:
            raise ValueError("A must be a one-dimensional vector.")

        if not np.issubdtype(A.dtype, np.number):
            raise TypeError("A must contain numeric values.")

        if not np.all(np.isin(A, [0, 1])):
            raise ValueError("A must only contain binary values 0 and 1.")

        return A.astype(int)
    
    @staticmethod
    def _split_sample(X: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Split a feature matrix into two group-specific feature matrices.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.
        A : np.ndarray
            Sensitive variable vector, encoded as 0 and 1.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Feature matrices X_0 and X_1, where X_0 contains the observations
            with A = 0 and X_1 contains the observations with A = 1.
        """
        X = FairPCA._validate_features(X)
        A = FairPCA._validate_sensitives(A)

        if X.shape[0] != A.shape[0]:
            raise ValueError("X and A must contain the same number of observations.")

        X_0 = X[A == 0]
        X_1 = X[A == 1]

        return X_0, X_1
        
    
    def _check_is_fitted(self) -> None:
        """Check whether the Fair PCA instance has been fitted."""
        if not self.fitted:
            raise RuntimeError("The Fair PCA model must be fitted before this method is used.")