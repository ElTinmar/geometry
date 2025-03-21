import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union

def homogeneous(X: NDArray, z: float = 0):

    if X.ndim != 2:
        raise ValueError(f'X must be a 2D array')
    
    shape_homogeneous = (X.shape[0], X.shape[1]+1)
    X_homogeneous = np.empty(shape_homogeneous, dtype=X.dtype)
    X_homogeneous[:,:-1] = X
    X_homogeneous[:,-1] = z

    return X_homogeneous

def homogeneous_2d(X: NDArray, z: float = 0):
    X = np.atleast_2d(X)
    if X.shape[1] != 2:
        raise ValueError(f'Expected input shape (2,) or (N,2), but got {X.shape}')
    return homogeneous(X, z)

def homogeneous_coord_2d(X: NDArray):
    return homogeneous_2d(X, 1)

def homogeneous_vec_2d(X: NDArray):
    return homogeneous_2d(X, 0)
    
class AffineTransform2D(np.ndarray):

    def __new__(cls) -> "AffineTransform2D":
        return np.eye(3, dtype=np.float32).view(cls)

    def __array_finalize__(self, obj):
        """Ensure attributes are inherited when slicing or viewing."""
        if obj is None:
            return

    @classmethod
    def from_array(cls, input_array: NDArray) -> "AffineTransform2D":
        # safe method that checks that the input array represents
        # an invertible affine transformation

        obj = np.asarray(input_array, dtype=np.float32).view(cls)

        if obj.shape != (3, 3):
            raise ValueError("AffineTransform2D must be a 3x3 matrix.")
        
        if not np.array_equal(obj[2], [0, 0, 1]):
            raise ValueError("Last row should be [0, 0, 1]")
        
        linear_part = obj[:2, :2]
        det = np.linalg.det(linear_part)
        if np.isclose(det, 0):
            raise ValueError("Transform should be invertible")
        
        return obj

    @classmethod
    def _from_array(cls, input_array: NDArray) -> "AffineTransform2D":
        # this method is only to be used internally and trusts that 
        # the input array is well behaved 

        obj = np.asarray(input_array, dtype=np.float32).view(cls)
        if obj.shape != (3, 3):
            raise ValueError("AffineTransform2D must be a 3x3 matrix.")
        
        return obj
        
    def transform_points(self, points_2d: NDArray) -> NDArray:

        x_homogeneous = homogeneous_coord_2d(points_2d)
        x_transformed = x_homogeneous @ np.asarray(self).T
        return x_transformed[:,:-1]

    def transform_vectors(self, vectors_2d: NDArray) -> NDArray:
        
        x_homogeneous = homogeneous_vec_2d(vectors_2d)
        x_transformed = x_homogeneous @ np.asarray(self).T
        return x_transformed[:,:-1]
    
    def __matmul__(self, other: Union["AffineTransform2D", np.ndarray]) ->  Union["AffineTransform2D",np.ndarray]:
        
        result = np.matmul(self, other)
        
        if isinstance(other, AffineTransform2D):
            return AffineTransform2D._from_array(result)
        
        elif isinstance(other, np.ndarray):
            return np.asarray(result)
        
        else:
            return NotImplemented

    def __rmatmul__(self, other: Union["AffineTransform2D", np.ndarray]) ->  Union["AffineTransform2D",np.ndarray]:

        result = np.matmul(other, self)

        if isinstance(other, AffineTransform2D):
            return AffineTransform2D._from_array(result)
        
        elif isinstance(other, np.ndarray):
            return np.asarray(result)
        
        else:
            return NotImplemented
        
    @classmethod
    def identity(cls) -> "AffineTransform2D":
        I = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],        
        ])
        return cls._from_array(I)

    @classmethod
    def translation(cls, tx: float, ty: float) -> "AffineTransform2D":
        T = np.array([
            [1.0, 0.0,  tx],
            [0.0, 1.0,  ty],
            [0.0, 0.0, 1.0],        
        ])
        return cls._from_array(T)

    @classmethod
    def rotation(cls, angle_rad: float) -> "AffineTransform2D":
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        R = np.array([
            [c,    -s, 0.0],
            [s,     c, 0.0],
            [0.0, 0.0, 1.0]
        ])
        return cls._from_array(R)

    @classmethod
    def scaling(cls, sx: float, sy: Optional[float] = None) -> "AffineTransform2D":
        
        if sy is None:
            sy = sx

        S = np.array([
            [ sx, 0.0, 0.0],
            [0.0,  sy, 0.0],
            [0.0, 0.0, 1.0]
        ])
        return cls._from_array(S)

    def translate(self, tx: float, ty: float) -> "AffineTransform2D":
        return AffineTransform2D.translation(tx, ty) @ self

    def scale(self, sx: float, sy: Optional[float] = None) -> "AffineTransform2D":
        return AffineTransform2D.scaling(sx, sy) @ self

    def rotate(self, angle_rad: float) -> "AffineTransform2D":
        return AffineTransform2D.rotation(angle_rad) @ self

    def inv(self) -> "AffineTransform2D":
        return AffineTransform2D._from_array(np.linalg.inv(self))

class SimilarityTransform2D(AffineTransform2D):

    @classmethod
    def from_array(cls, input_array: NDArray) -> "AffineTransform2D":
        # safe method that checks that the input array represents
        # an invertible affine transformation

        obj = np.asarray(input_array, dtype=np.float32).view(cls)

        if obj.shape != (3, 3):
            raise ValueError("Transform must be a 3x3 matrix.")
        
        if not np.array_equal(obj[2], [0, 0, 1]):
            raise ValueError("Last row should be [0, 0, 1]")
        
        linear_part = obj[:2, :2]
        det = np.linalg.det(linear_part)
        if np.isclose(det, 0):
            raise ValueError("Transform should be invertible")
        
        dot_product = np.dot(linear_part[:, 0], linear_part[:, 1])
        if not np.isclose(dot_product, 0):
            raise ValueError("The columns of the linear part must be orthogonal.")

        norm_col0 = np.linalg.norm(linear_part[:, 0])
        norm_col1 = np.linalg.norm(linear_part[:, 1])
        if not np.isclose(norm_col0, norm_col1):
            raise ValueError("The columns of the linear part must have equal norms.")
        
        return obj
    
    def __matmul__(self, other: Union["SimilarityTransform2D", "AffineTransform2D", np.ndarray]) ->  Union["SimilarityTransform2D","AffineTransform2D",np.ndarray]:
        
        result = np.matmul(self, other)

        if isinstance(other, SimilarityTransform2D):
            return SimilarityTransform2D._from_array(result) 
        
        if isinstance(other, AffineTransform2D):
            return AffineTransform2D._from_array(result) 
    
        if isinstance(other, np.ndarray):
            return np.asarray(result)
        
        return NotImplemented
        
    def __rmatmul__(self, other: Union["SimilarityTransform2D", "AffineTransform2D", np.ndarray]) -> Union["SimilarityTransform2D","AffineTransform2D",np.ndarray]:
        
        result = np.matmul(other, self)

        if isinstance(other, SimilarityTransform2D):
            return SimilarityTransform2D._from_array(result) 
        
        if isinstance(other, AffineTransform2D):
            return AffineTransform2D._from_array(result) 
        
        if isinstance(other, np.ndarray):
            return np.asarray(result)
        
        return NotImplemented

    @classmethod
    def identity(cls) -> "SimilarityTransform2D":
        """Create an identity similarity transformation."""
        return cls._from_array(AffineTransform2D.identity())

    @classmethod
    def translation(cls, tx: float, ty: float) -> "SimilarityTransform2D":
        """Create a translation similarity transformation."""
        return cls._from_array(AffineTransform2D.translation(tx, ty))

    @classmethod
    def rotation(cls, angle_rad: float) -> "SimilarityTransform2D":
        """Create a rotation similarity transformation."""
        return cls._from_array(AffineTransform2D.rotation(angle_rad))

    @classmethod
    def scaling(cls, s: float) -> "SimilarityTransform2D":
        """Create a uniform scaling similarity transformation."""
        return cls._from_array(AffineTransform2D.scaling(s))

    def translate(self, tx: float, ty: float) -> "SimilarityTransform2D":
        return SimilarityTransform2D.translation(tx, ty) @ self

    def scale(self, s: float) -> "SimilarityTransform2D":
        return SimilarityTransform2D.scaling(s) @ self

    def rotate(self, angle_rad: float) -> "SimilarityTransform2D":
        return SimilarityTransform2D.rotation(angle_rad) @ self
    
    def inv(self) -> "SimilarityTransform2D":
        """Return the inverse of the similarity transformation."""
        return SimilarityTransform2D._from_array(np.linalg.inv(self))
    
    @property
    def scale_factor(self) -> float:
        return np.linalg.norm(self[:,0])
        

