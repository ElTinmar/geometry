import timeit

setup_code = """
from geometry import pca
import numpy as np
from sklearn.decomposition import PCA

mean = [3, 10]
cov = [[10, 25], [25, 100]]
X = np.random.multivariate_normal(mean, cov, 1000)

def pca_sk(X):
    pca = PCA()
    scores = pca.fit_transform(X)     
    return pca.components_, scores
"""

sklearn = "pca_sk(X)"
simple_pca = "pca(X)"

N = 10_000
t_simple_pca = timeit.timeit(stmt=simple_pca, setup=setup_code, number=N)
t_sklearn = timeit.timeit(stmt=sklearn, setup=setup_code, number=N)

print(f'simple {t_simple_pca}s, sklearn PCA {t_sklearn}s, speedup {t_sklearn/t_simple_pca:0.2f} X')
