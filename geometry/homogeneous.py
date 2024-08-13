import numpy as np
from numpy.linalg import inv
from numpy.typing import NDArray, ArrayLike
from vector import row_to_col, col_to_row

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
    if len(shp) == 1:
        homogeneous_coords = np.append(input_coords,1)
    else:
        new_colum = np.ones((n,1))
        homogeneous_coords = np.hstack((input_coords, new_colum))
    
    return homogeneous_coords

def from_homogeneous(homogeneous_coords: ArrayLike) -> NDArray:

    # transform to numpy array
    homogeneous_coords = np.asarray(homogeneous_coords)

    return homogeneous_coords[:,:-1]    

class Affine2DTransform():
    
    @staticmethod
    def identity() -> NDArray:
        I = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],        
        ])
        return I

    @staticmethod
    def translation(tx: float, ty: float) -> NDArray:
        T = np.array([
            [1.0, 0.0,  tx],
            [0.0, 1.0,  ty],
            [0.0, 0.0, 1.0],        
        ])
        return T

    @staticmethod
    def rotation(angle_rad: float) -> NDArray:
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        R = np.array([
            [c,    -s, 0.0],
            [s,     c, 0.0],
            [0.0, 0.0, 1.0]
        ])
        return R

    @staticmethod
    def scaling(sx: float, sy:float) -> NDArray:
        S = np.array([
            [ sx, 0.0, 0.0],
            [0.0,  sy, 0.0],
            [0.0, 0.0, 1.0]
        ])
        return S

    @staticmethod
    def inverse(T: NDArray) -> NDArray:
        return inv(T)
    
def transform2d(T: Affine2DTransform, x: NDArray):
    '''
    input: x is an array with shape (2,) , (1,2) or (N,2)
    output: returns an array with shape (1,2) or (N,2) 
    '''
    
    shp = x.shape

    if len(shp) == 1 and shp[0] == 2:
        # make it explicitly a 1x2 row vector 
        x = x[np.newaxis,:]

    elif len(shp) == 2 and shp[1] == 2:
        # Nx2 array, nothing to do
        pass

    else:
        return ValueError('Wrong shape, please provide an array with shape (2,) , (1,2) or (N,2)')

    x_homogeneous = to_homogeneous(x)
    y_homogeneous = T @ x_homogeneous.T
    return from_homogeneous(y_homogeneous.T)

