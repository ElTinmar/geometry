import timeit

setup_code = """
from geometry import pca
import numpy as np
from sklearn.decomposition import PCA

mean = [0, 0]
cov = [[1, 0], [0, 100]]
X = np.random.multivariate_normal(mean, cov, 100)

def pca_sk(X):
    pca = PCA()
    scores = pca.fit_transform(X)     
    return pca.components_
"""

sklearn = "pca_sk(X)"
simple_pca = "pca(X)"

N = 10_000
t_simple_pca = timeit.timeit(stmt=simple_pca, setup=setup_code, number=N)
t_sklearn = timeit.timeit(stmt=sklearn, setup=setup_code, number=N)

print(f'simple {t_simple_pca}s, sklearn PCA {t_sklearn}s, speedup {t_sklearn/t_simple_pca:0.2f} X')
