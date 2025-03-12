import numpy as np
from numpy.typing import NDArray

def pca(X: NDArray):
    '''return PCs on rows'''

    X_centered = X - np.mean(X, axis=0)
    cov = X_centered.T @ X_centered
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    return eigenvectors[:, ::-1].T

