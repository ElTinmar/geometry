import numpy as np
from numpy.linalg import inv
from numpy.typing import NDArray, ArrayLike
from typing import Optional

def to_homogeneous_vector(input_vector: ArrayLike) -> NDArray:
    '''
    input_coords : NxD array where N is the number of points and D the number of dimensions
    homogeneous_coords: Nx(D+1) homogeneous coordinates
    '''
    
    # transform to numpy array
    input_vector = np.asarray(input_vector)

    # check shape
    shp = input_vector.shape    
    if len(shp) > 2:
        raise ValueError('Expected NxD array')
    
    # add a column of ones
    n = shp[0]
    if len(shp) == 1:
        homogeneous_coords = np.append(input_vector,0)
    else:
        new_colum = np.zeros((n,1))
        homogeneous_coords = np.hstack((input_vector, new_colum))
    
    return homogeneous_coords

def to_homogeneous_point(input_coords: ArrayLike) -> NDArray:
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

def from_homogeneous(homogeneous: ArrayLike) -> NDArray:

    # transform to numpy array
    homogeneous = np.asarray(homogeneous)

    return homogeneous[:,:-1]    

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
    def scaling(sx: float, sy: Optional[float] = None) -> NDArray:
        
        if sy is None:
            sy = sx

        S = np.array([
            [ sx, 0.0, 0.0],
            [0.0,  sy, 0.0],
            [0.0, 0.0, 1.0]
        ])
        return S

    @staticmethod
    def inverse(T: NDArray) -> NDArray:
        return inv(T)
    
def transform_point_2d(T: Affine2DTransform, x: NDArray):
    '''
    input: x is an array with shape (2,) , (1,2) or (N,2)
    output: returns an array with shape (1,2) or (N,2) 
    '''
    
    if x.shape == (2,):
        x = x[np.newaxis, :]  # Convert (2,) to (1,2)

    elif x.ndim != 2 or x.shape[1] != 2:
        raise ValueError('Expected input shape (2,), (1,2), or (N,2), but got {}'.format(x.shape))

    x_homogeneous = to_homogeneous_point(x)
    y_homogeneous = T @ x_homogeneous.T
    return from_homogeneous(y_homogeneous.T)

def transform_vector_2d(T: Affine2DTransform, v: NDArray):
    '''
    input: x is an array with shape (2,) , (1,2) or (N,2)
    output: returns an array with shape (1,2) or (N,2) 
    '''
    
    if v.shape == (2,):
        v = v[np.newaxis, :]  # Convert (2,) to (1,2)

    elif v.ndim != 2 or v.shape[1] != 2:
        raise ValueError('Expected input shape (2,), (1,2), or (N,2), but got {}'.format(v.shape))

    x_homogeneous = to_homogeneous_vector(v)
    y_homogeneous = T @ x_homogeneous.T
    return from_homogeneous(y_homogeneous.T)
