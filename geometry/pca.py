import numpy as np
from numpy.typing import NDArray

def pca(X: NDArray):

    X_centered = X - np.mean(X, axis=0)
    cov = X_centered.T @ X_centered
    _, eigenvectors = np.linalg.eigh(cov)
    components = eigenvectors[:, ::-1]
    scores = X_centered @ components
    return components, scores

