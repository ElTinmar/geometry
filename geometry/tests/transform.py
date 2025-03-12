
import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from geometry.transform2d import *

class TestAffineTransform2D(unittest.TestCase):

    def setUp(self):
        self.T = AffineTransform2D.translation(3, 4)
        self.R = AffineTransform2D.rotation(np.pi / 4)  
        self.S = AffineTransform2D.scaling(2)
        self.I = AffineTransform2D.identity()

    def test_identity(self):
        I = AffineTransform2D.identity()
        expected = np.eye(3)
        assert_array_almost_equal(I, expected)
    
    def test_translation(self):
        T = AffineTransform2D.translation(3, 4)
        expected = np.array([
            [1.0, 0.0, 3.0],
            [0.0, 1.0, 4.0],
            [0.0, 0.0, 1.0]
        ])
        assert_array_almost_equal(T, expected)
    
    def test_rotation(self):
        angle = np.pi / 4  # 45 degrees
        R = AffineTransform2D.rotation(angle)
        c, s = np.cos(angle), np.sin(angle)
        expected = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ])
        assert_array_almost_equal(R, expected)
    
    def test_scaling(self):
        S = AffineTransform2D.scaling(2, 3)
        expected = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        assert_array_almost_equal(S, expected)
    
    def test_transform_points(self):
        T = AffineTransform2D.translation(1, 2)
        points = np.array([[3, 4], [5, 6]])
        transformed = T.transform_points(points)
        expected = np.array([[4, 6], [6, 8]])
        assert_array_almost_equal(transformed, expected)
    
    def test_transform_vectors(self):
        S = AffineTransform2D.scaling(2, 3)
        vectors = np.array([[3, 4], [5, 6]])
        transformed = S.transform_vectors(vectors)
        expected = np.array([[6, 12], [10, 18]])
        assert_array_almost_equal(transformed, expected)
    
    def test_inverse(self):
        T = AffineTransform2D.translation(3, 4)
        T_inv = T.inv()
        expected = AffineTransform2D.translation(-3, -4)
        assert_array_almost_equal(T_inv, expected)
    
    def test_is_similarity(self):
        S = AffineTransform2D.scaling(2) 
        R = AffineTransform2D.rotation(np.pi / 4)  
        T = AffineTransform2D.translation(1, 1) 
        self.assertTrue(S.is_similarity())
        self.assertTrue(R.is_similarity())
        self.assertTrue(T.is_similarity())  
        non_sim = AffineTransform2D.scaling(2, 3) 
        self.assertFalse(non_sim.is_similarity())

    def test_matmul_affine(self):
        """Test AffineTransform2D @ AffineTransform2D"""
        result = self.T @ self.R
        expected = np.matmul(self.T, self.R)
        np.testing.assert_array_almost_equal(result, expected)
        self.assertIsInstance(result, AffineTransform2D)

    def test_matmul_ndarray(self):
        """Test AffineTransform2D @ np.ndarray"""
        array = np.eye(3)
        result = self.T @ array
        expected = np.matmul(self.T, array)
        np.testing.assert_array_almost_equal(result, expected)
        self.assertIsInstance(result, np.ndarray)  
        self.assertNotIsInstance(result, AffineTransform2D)  

    def test_rmatmul_ndarray(self):
        """Test np.ndarray @ AffineTransform2D"""
        array = np.eye(3)
        result = array @ self.T
        expected = np.matmul(array, self.T)
        np.testing.assert_array_almost_equal(result, expected)
        self.assertIsInstance(result, np.ndarray)  
        self.assertNotIsInstance(result, AffineTransform2D) 

    def test_invalid_matmul(self):
        """Ensure AffineTransform2D @ invalid type raises an error"""
        with self.assertRaises(TypeError):
            self.T @ "invalid"

    def test_invalid_rmatmul(self):
        """Ensure np.ndarray @ invalid type raises an error"""
        with self.assertRaises(TypeError):
            "invalid" @ self.T
    
    def test_chaining_operations(self):
        """Test if chained affine transformations are applied correctly."""
        
        S = AffineTransform2D.scaling(2) 
        T = AffineTransform2D.translation(1, 2)
        expected = T @ S  
        result = AffineTransform2D().scale(2).translate(1, 2)
        np.testing.assert_array_almost_equal(result, expected)

class TestSimilarityTransform2D(unittest.TestCase):
    
    def test_identity(self):
        I = SimilarityTransform2D.identity()
        expected = np.eye(3)
        assert_array_almost_equal(I, expected)
    
    def test_translation(self):
        T = SimilarityTransform2D.translation(3, 4)
        expected = AffineTransform2D.translation(3, 4)
        assert_array_almost_equal(T, expected)
    
    def test_rotation(self):
        angle = np.pi / 6
        R = SimilarityTransform2D.rotation(angle)
        expected = AffineTransform2D.rotation(angle)
        assert_array_almost_equal(R, expected)
    
    def test_scaling(self):
        S = SimilarityTransform2D.scaling(2)
        expected = AffineTransform2D.scaling(2, 2)
        assert_array_almost_equal(S, expected)
    
    def test_inverse(self):
        T = SimilarityTransform2D.translation(3, 4)
        T_inv = T.inv()
        expected = SimilarityTransform2D.translation(-3, -4)
        assert_array_almost_equal(T_inv, expected)
    
    def test_invalid_similarity(self):
        with self.assertRaises(ValueError):
            SimilarityTransform2D(AffineTransform2D.scaling(2, 3))
    
    def test_scale_factor(self):
        S = SimilarityTransform2D.scaling(2)
        self.assertAlmostEqual(S.scale_factor, 2.0)
        
        R = SimilarityTransform2D.rotation(np.pi / 4)
        self.assertAlmostEqual(R.scale_factor, 1.0)

    def test_similarity_times_similarity(self):
        R = SimilarityTransform2D.rotation(np.pi / 4)
        S = SimilarityTransform2D.scaling(2)
        T = SimilarityTransform2D.translation(3, 4)

        result = R @ S @ T
        self.assertIsInstance(result, SimilarityTransform2D)
        self.assertTrue(result.is_similarity())

    def test_similarity_times_affine_non_similarity(self):
        R = SimilarityTransform2D.rotation(np.pi / 4)
        A = AffineTransform2D.scaling(2, 3)  

        result1 = R @ A
        self.assertIsInstance(result1, AffineTransform2D)
        self.assertNotIsInstance(result1, SimilarityTransform2D)
        self.assertFalse(result1.is_similarity())

        result2 = A @ R
        self.assertIsInstance(result2, AffineTransform2D)
        self.assertNotIsInstance(result2, SimilarityTransform2D)
        self.assertFalse(result2.is_similarity())

    def test_similarity_times_affine_similarity(self):
        R = SimilarityTransform2D.rotation(np.pi / 4)
        S = AffineTransform2D.scaling(2, 2) 

        result1 = R @ S
        self.assertIsInstance(result1, AffineTransform2D)
        self.assertTrue(result1.is_similarity())

        result2 = S @ R
        self.assertIsInstance(result2, AffineTransform2D)
        self.assertTrue(result2.is_similarity())

    def test_affine_non_similarity_times_affine_similarity(self):
        A = AffineTransform2D.scaling(2, 3)  
        S = AffineTransform2D.scaling(2, 2)  

        result = A @ S
        self.assertIsInstance(result, AffineTransform2D)
        self.assertNotIsInstance(result, SimilarityTransform2D)
        self.assertFalse(result.is_similarity())

        result2 = S @ A
        self.assertIsInstance(result2, AffineTransform2D)
        self.assertNotIsInstance(result2, SimilarityTransform2D)
        self.assertFalse(result2.is_similarity())

    def test_similarity_times_ndarray(self):
        R = SimilarityTransform2D.rotation(np.pi / 4)
        M = np.array([
            [1.5, 0.0, 1.0],
            [0.0, 1.5, 2.0],
            [0.0, 0.0, 1.0]
        ])

        result = R @ M  
        self.assertIsInstance(result, np.ndarray)
        self.assertNotIsInstance(result, SimilarityTransform2D)
        self.assertNotIsInstance(result, AffineTransform2D)

        result2 = M @ R  
        self.assertIsInstance(result2, np.ndarray)
        self.assertNotIsInstance(result, SimilarityTransform2D)
        self.assertNotIsInstance(result2, AffineTransform2D)

    def test_chaining_operations(self):
        """Test if chained affine transformations are applied correctly."""
        
        S = SimilarityTransform2D.scaling(2) 
        T = SimilarityTransform2D.translation(1, 2)
        expected = T @ S  
        result = SimilarityTransform2D().scale(2).translate(1, 2)
        np.testing.assert_array_almost_equal(result, expected)
        self.assertIsInstance(result, SimilarityTransform2D)

if __name__ == "__main__":
    unittest.main()
