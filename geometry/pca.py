import numpy as np
from numpy.typing import NDArray

def pca(X: NDArray):
    mu = np.mean(X, axis=0)
    X_centered = X - mu
    cov = X_centered.T @ X_centered
    _, eigenvectors = np.linalg.eigh(cov)
    components = eigenvectors[:, ::-1]
    scores = X_centered @ components
    return mu, components, scores

