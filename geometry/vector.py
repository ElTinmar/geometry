import numpy as np
from numpy.typing import NDArray

def col_to_row(col: NDArray) -> NDArray:
    
    shp = col.shape
    
    if (len(shp) > 2) or (shp[1] > 1):
        raise ValueError('not a column vector')
    
    return col.reshape(1, -1)
    
def row_to_col(row: NDArray) -> NDArray:
    
    shp = row.shape
    
    if (len(shp) > 2) or (shp[0] > 1):
        raise ValueError('not a row vector')
    
    return row.reshape(-1, 1)
    