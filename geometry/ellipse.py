import numpy as np
from numpy.typing import NDArray

def ellipse_direction(inertia_tensor: NDArray, heading: NDArray) -> NDArray:
    '''
    Get ellipse orientation: return the first eigenvector 
    of the inertia tensor, which corresponds to the principal 
    axis of the ellipse.
    Resolve 180 deg ambiguity by aligning with heading vector
    '''
    eigvals, eigvecs = np.linalg.eig(inertia_tensor)
    loc = np.argmax(abs(eigvals))
    dir = eigvecs[loc,:]
    # solve 180 deg ambiguity
    if dir @ heading < 0:
        dir = -dir
    return dir