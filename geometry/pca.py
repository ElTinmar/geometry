import numpy as np
from numpy.typing import NDArray

def pca(X: NDArray):

    X_centered = X - np.mean(X, axis=0)
    cov = X_centered.T @ X_centered
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    ord = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:,ord]