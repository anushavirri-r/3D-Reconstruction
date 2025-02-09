import numpy as np


def read_matrix(path, astype=np.float64):
    """
    parameters:: path: path to the file, astype: type to cast the numbers. Default value: np.float64
    """
    with open(path, 'r') as f:
        # Reads a file containing a matrix where each line represents a point and each point is tab or space separated.
        arr = []
        for line in f:
            arr.append([(token if token != '*' else -1) # appends the file lines by spliting at each point into the array "arr" and replaces * with -1
                        for token in line.strip().split()])
        # returns array of array of numbers
        return np.asarray(arr).astype(astype)

# Convert cartesian to homogenous points by appending a row of 1s
def cart2hom(arr):
    """ 
    parameters: arr
    """
    # arr shape: dimensions x number_points
    if arr.ndim == 1:
        return np.hstack([arr, 1])
    # returns array of shape ((num_dimension+1) x num_points)
    return np.asarray(np.vstack([arr, np.ones(arr.shape[1])]))

# Convert homogenous to cartesian by dividing each row by the last row
def hom2cart(arr):
    """ 
    parameters: arr
    """
    # arr shape: dimensions x number_points
    num_rows = len(arr)
    if num_rows == 1 or arr.ndim == 1:
        return arr
    # returns array of shape((num_dimension-1) x num_points) iff d > 1
    return np.asarray(arr[:num_rows - 1] / arr[num_rows - 1])
