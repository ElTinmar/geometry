import unittest
from geometry import to_homogeneous, from_homogeneous
import numpy as np
from numpy.testing import assert_array_equal

class TestHomogeneous(unittest.TestCase):

    def test_to_homogeneous(self):
        input = np.array([[1,2]])
        output = np.array([[1,2,1]])
        assert_array_equal(output, to_homogeneous(input))
    
    def test_from_homogeneous(self):
        input = np.array([[1,2,1]])
        output = np.array([[1,2]])
        assert_array_equal(output, from_homogeneous(input))

if __name__ == '__main__':
    unittest.main()