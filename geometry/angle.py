import numpy as np
from numpy.typing import NDArray

def angle_between_vectors(v1: NDArray, v2: NDArray, signed: bool = True) -> float:
    v1_unit = v1 / np.linalg.norm(v1)
    v2_unit = v2 / np.linalg.norm(v2)
    signed_angle = np.arccos(np.dot(v1_unit,v2_unit))
    if signed:
        if np.cross(v1_unit,v2_unit) < 0:
            signed_angle = -signed_angle
    return signed_angle

def angdiff(a1: float, a2: float) -> float: 
    angle_diff = a1 - a2
    angle_diff = ((angle_diff + np.pi) % (2*np.pi)) - np.pi
    return angle_diff

def normalize_angle(theta: float) -> float:
    '''returns angle in [-pi, pi]'''
    return np.arctan2(np.sin(theta), np.cos(theta))

'''
def angular_difference(a, b):
    #signed angular difference
    return np.arctan2(np.sin(a - b), np.cos(a - b))

def angle_between_vectors(v1: NDArray, v2: NDArray) -> float:
    return np.arctan2(np.cross(v1,v2), np.dot(v1,v2))
'''