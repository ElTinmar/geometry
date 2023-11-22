import numpy as np
from numpy.typing import NDArray, ArrayLike

def to_homogeneous(input_coords: ArrayLike) -> NDArray:
    '''
    input_coords : NxD array where N is the number of points and D the number of dimensions
    homogeneous_coords: Nx(D+1) homogeneous coordinates
    '''
    
    # transform to numpy array
    input_coords = np.asarray(input_coords)

    # check shape
    shp = input_coords.shape    
    if len(shp) > 2:
        raise ValueError('Expected NxD array')
    
    # add a column of ones
    n = shp[0]
    new_colum = np.ones((n,1))
    homogeneous_coords = np.hstack((input_coords, new_colum))
    
    return homogeneous_coords

def from_homogeneous(homogeneous_coords: ArrayLike) -> NDArray:

    # transform to numpy array
    homogeneous_coords = np.asarray(homogeneous_coords)

    return homogeneous_coords[:,:-1]
