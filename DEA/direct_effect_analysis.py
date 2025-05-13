import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from scipy.linalg import null_space
from scipy.stats import f
from tqdm import tqdm

class DirectEffectAnalysis(BaseEstimator, TransformerMixin):
    """
    A class for analyzing the direct effect of Z on Y, while controlling for X.

    Parameters
    ----------
    n_components : int or 'optimal', default=50
        Number of principal components to retain in PCA. If 'optimal', selects the best number
        using k-fold cross-validation.
    
    alpha : float, default=1e-6
        Regularization parameter for inverting the noise covariance matrix.
    
    k_fold : int, default=5
        Number of folds for cross-validation when selecting the optimal number of components.

    n_cps : int, default=np.logspace(0.1, 3, 20).astype('int')
        List of number of PCs to be teste if n_components='optimal'.
    """
    def __init__(self, n_components='50', alpha=1e-6, k_fold=5, n_cps=np.logspace(0.1, 3, 20).astype('int')):
        self.n_components = n_components
        self.alpha = alpha
        self.k_fold = k_fold
        self.n_cps = n_cps
        
    def select_n_component(self, X, Y, Z):
        """
        Selects the optimal number of principal components using k-fold cross-validation
        based on the R² score of a PCA + regression model.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.
        
        Y : ndarray of shape (n_samples, n_targets)
            Target variables.
        
        Z : ndarray of shape (n_samples, n_covariates)
            Covariates for testing direct effects.
        """
        best_score = -np.inf
        best_n = 1
        
        kf = KFold(n_splits=self.k_fold, shuffle=True, random_state=42)
        
        for n in tqdm(self.n_cps):
            scores = []
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                Y_train, Y_val = Y[train_idx], Y[val_idx]
                Z_train, Z_val = Z[train_idx], Z[val_idx]
                
                pca = PCA(n_components=n)
                X_train_pca = pca.fit_transform(X_train)
                X_val_pca = pca.transform(X_val)
                
                lr = LinearRegression()
                lr.fit(np.hstack((X_train_pca, Z_train)), Y_train)
                
                scores.append(lr.score(np.hstack((X_val_pca, Z_val)), Y_val))
            
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_n = n
        
        self.n_components_ = best_n
        
    def fit(self, X, Y, Z, fit_test=True):
        """
        Fit the model and optionally test whether Z directly causes Y.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.
        
        Y : ndarray of shape (n_samples, n_targets)
            Target variables.
        
        Z : ndarray of shape (n_samples, n_covariates)
            Covariates for testing direct effects.
        
        fit_test : bool, default=True
            If True, performs a statistical test for direct causation.

        Returns
        -------
        stat : float
            Test statistic (if fit_test is True, else None).
        
        pval : float
            p-value for the statistical test (if fit_test is True, else None).
        """
        N, p, r, d = X.shape[0], X.shape[1], Z.shape[1], Y.shape[1]
        
        # Determine optimal number of components if needed
        if self.n_components == 'optimal':
            self.select_n_component(X, Y, Z)
        else:
            self.n_components_ = self.n_components
        
        # Apply PCA to X
        self.pca_ = PCA(n_components=self.n_components_)
        X_pca = self.pca_.fit_transform(X)
        
        # Fit linear regression model
        self.lr_ = LinearRegression()
        self.lr_.fit(np.hstack((X_pca, Z)), Y)
        
        # Compute regression coefficients and null space
        self.B_ = self.lr_.coef_[:, -1, None]  # Last column corresponds to Z
        self.B_perp_ = null_space(self.B_.T)
        self.P_dyn_ = self.B_perp_ @ np.linalg.inv(self.B_perp_.T @ self.B_perp_) @ self.B_perp_.T
        
        # Compute noise covariance
        residuals = Y - self.lr_.predict(np.hstack((np.zeros(X_pca.shape), Z)))
        self.Sigma_ = np.cov(residuals, rowvar=False)
        
        # Compute transformation coefficients
        self.coef_ = np.linalg.inv(self.Sigma_ + self.alpha * np.eye(self.Sigma_.shape[0])) @ self.B_
        
        # Fit statistical test for Z -> Y|X
        if fit_test:
            eigenvalues, _ = np.linalg.eigh(self.coef_ @ self.B_.T)
            df1, df2 = d, N - p - r - d - 1
            stat = (df2 / df1) * np.max(eigenvalues)
            pval = f.sf(stat, df1, df2)
            return stat, pval
        
        return None, None
    
    def disentangle(self, Y, Z=None):
        """
        Decomposes Y into components directly caused by Z and those that are orthogonal.
        
        Parameters
        ----------
        Y : ndarray of shape (n_samples, n_targets)
            Target variable matrix.

        Z : ndarray of shape (n_samples, n_covariates)
            Covariates for testing direct effects.
        
        Returns
        -------
        Y_perp : ndarray of shape (n_samples, n_targets)
            The component of Y orthogonal to Z.
        
        Y_dir : ndarray of shape (n_samples, n_targets)
            The component of Y directly caused by Z.
        """
        Y_perp = Y @ self.P_dyn_
        Y_dir = Y @ self.coef_ @ self.B_.T
        return Y_perp, Y_dir
    

    def counterfactual(self, Y, Z):
        """
        Separate the direct effect and perpendicular effects of Y for a counterfactual value of Z.
        
        Parameters
        ----------
        Y : ndarray of shape (n_samples, n_targets)
            Target variable matrix.

        Z : ndarray of shape (n_samples, n_covariates)
            Covariates for testing direct effects.
        
        Returns
        -------
        Y_perp : ndarray of shape (n_samples, n_targets)
            The component of Y orthogonal to Z.
        
        Y_dir : ndarray of shape (n_samples, n_targets)
            The component of Y directly caused by Z.
        """
        Y_perp = Y @ self.P_dyn_
        Y_dir = Z @ self.B_.T
        return Y_perp, Y_dir
    

    
    def transform(self, Y):
        """
        Projects Y into the optimal space based on estimated coefficients.
        
        Parameters
        ----------
        Y : ndarray of shape (n_samples, n_targets)
            Target variable matrix.
        
        Returns
        -------
        Y_transformed : ndarray of shape (n_samples, 1)
            Transformed representation of Y.
        """
        return Y @ self.coef_
