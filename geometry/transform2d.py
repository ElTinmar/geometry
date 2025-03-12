import numpy as np
from numpy.typing import NDArray
from typing import Optional, Callable, Union

class AffineTransform2D(np.ndarray):

    def __new__(cls, input_array: Optional[NDArray] = None) -> "AffineTransform2D":

        if input_array is None:
            input_array = np.eye(3, dtype=np.float64) 

        obj = np.asarray(input_array, dtype=np.float64).view(cls)
        if obj.shape != (3, 3):
            raise ValueError("AffineTransform2D must be a 3x3 matrix.")
        
        return obj

    def __array_finalize__(self, obj):
        """Ensure attributes are inherited when slicing or viewing."""
        if obj is None:
            return

    def _transform(self, x: NDArray, homogeneous_column: Callable[[int], NDArray]):

        x = np.atleast_2d(x)  
        
        if x.shape[1] != 2:
            raise ValueError(f'Expected input shape (2,) or (N,2), but got {x.shape}')

        x_homogeneous = np.column_stack((
            x, 
            homogeneous_column(x.shape[0])
        ))

        x_transformed = x_homogeneous @ self.T

        return x_transformed[:,:-1].squeeze()
        
    def transform_points(self, points_2d: NDArray):
        return self._transform(points_2d, lambda x: np.ones((x,1)))

    def transform_vectors(self, vectors_2d: NDArray):
        return self._transform(vectors_2d, lambda x: np.zeros((x,1)))
    
    def __matmul__(self, other: Union["AffineTransform2D", np.ndarray]) -> "AffineTransform2D":
        
        result = np.matmul(self, other)
        
        if isinstance(other, AffineTransform2D):
            return AffineTransform2D(result)
        
        elif isinstance(other, np.ndarray):
            return np.asarray(result)
        
        else:
            return NotImplemented

    def __rmatmul__(self, other: Union["AffineTransform2D", np.ndarray]) -> "AffineTransform2D":

        result = np.matmul(other, self)

        if isinstance(other, AffineTransform2D):
            return AffineTransform2D(result)
        
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
        return cls(I)

    @classmethod
    def translation(cls, tx: float, ty: float) -> "AffineTransform2D":
        T = np.array([
            [1.0, 0.0,  tx],
            [0.0, 1.0,  ty],
            [0.0, 0.0, 1.0],        
        ])
        return cls(T)

    @classmethod
    def rotation(cls, angle_rad: float) -> "AffineTransform2D":
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        R = np.array([
            [c,    -s, 0.0],
            [s,     c, 0.0],
            [0.0, 0.0, 1.0]
        ])
        return cls(R)

    @classmethod
    def scaling(cls, sx: float, sy: Optional[float] = None) -> "AffineTransform2D":
        
        if sy is None:
            sy = sx

        S = np.array([
            [ sx, 0.0, 0.0],
            [0.0,  sy, 0.0],
            [0.0, 0.0, 1.0]
        ])
        return cls(S)

    def translate(self, tx: float, ty: float) -> "AffineTransform2D":
        return AffineTransform2D.translation(tx, ty) @ self

    def scale(self, sx: float, sy: Optional[float] = None) -> "AffineTransform2D":
        return AffineTransform2D.scaling(sx, sy) @ self

    def rotate(self, angle_rad: float) -> "AffineTransform2D":
        return AffineTransform2D.rotation(angle_rad) @ self

    def inv(self) -> "AffineTransform2D":
        return AffineTransform2D(np.linalg.inv(self))
    
    def is_similarity(self) -> bool:
        """
        Check if the transformation is a similarity transform:
        rotation, translation, uniform scaling or reflection
        """
        U, S, Vt = np.linalg.svd(self[:2, :2]) 
        return np.allclose(S[0], S[1])

class SimilarityTransform2D(AffineTransform2D):
    
    def __new__(cls, input_array: Optional[NDArray] = None) -> "SimilarityTransform2D":
        obj = super().__new__(cls, input_array)
        if not obj.is_similarity():
            raise ValueError("Matrix is not a similarity transform.")
        return obj
    
    def __matmul__(self, other: Union["SimilarityTransform2D", "AffineTransform2D", np.ndarray]) -> "AffineTransform2D":
        
        result = np.matmul(self, other)

        if isinstance(other, SimilarityTransform2D):
            return SimilarityTransform2D(result) 
        
        if isinstance(other, AffineTransform2D):
            return AffineTransform2D(result) 
    
        if isinstance(other, np.ndarray):
            return np.asarray(result)
        
        return NotImplemented
        
    def __rmatmul__(self, other: Union["SimilarityTransform2D", "AffineTransform2D", np.ndarray]) -> "AffineTransform2D":
        
        result = np.matmul(other, self)

        if isinstance(other, SimilarityTransform2D):
            return SimilarityTransform2D(result) 
        
        if isinstance(other, AffineTransform2D):
            return AffineTransform2D(result) 
        
        if isinstance(other, np.ndarray):
            return np.asarray(result)
        
        return NotImplemented

    @classmethod
    def identity(cls) -> "SimilarityTransform2D":
        """Create an identity similarity transformation."""
        return cls(AffineTransform2D.identity())

    @classmethod
    def translation(cls, tx: float, ty: float) -> "SimilarityTransform2D":
        """Create a translation similarity transformation."""
        return cls(AffineTransform2D.translation(tx, ty))

    @classmethod
    def rotation(cls, angle_rad: float) -> "SimilarityTransform2D":
        """Create a rotation similarity transformation."""
        return cls(AffineTransform2D.rotation(angle_rad))

    @classmethod
    def scaling(cls, s: float) -> "SimilarityTransform2D":
        """Create a uniform scaling similarity transformation."""
        return cls(AffineTransform2D.scaling(s))

    def translate(self, tx: float, ty: float) -> "SimilarityTransform2D":
        return SimilarityTransform2D.translation(tx, ty) @ self

    def scale(self, s: float) -> "SimilarityTransform2D":
        return SimilarityTransform2D.scaling(s) @ self

    def rotate(self, angle_rad: float) -> "SimilarityTransform2D":
        return SimilarityTransform2D.rotation(angle_rad) @ self
    
    def inv(self) -> "SimilarityTransform2D":
        """Return the inverse of the similarity transformation."""
        return SimilarityTransform2D(super().inv())
    
    @property
    def scale_factor(self) -> float:
        U, S, Vt = np.linalg.svd(self[:2, :2])
        return S[0]
        

