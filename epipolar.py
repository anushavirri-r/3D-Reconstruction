import matplotlib.pyplot as plt
import numpy as np

def reconstruct_points(p1, p2, m1, m2):
    num_points = p1.shape[1]
    res = np.ones((4, num_points))

    for i in range(num_points):
        res[:, i] = reconstruct_one_point(p1[:, i], p2[:, i], m1, m2)

    return res


def skew(x):
    """ 
    :parameter x is a 3d vector
    """
    # Create a skew symmetric matrix *A* from a 3d vector *x*.
    # Property: np.cross(A, v) == np.dot(x, v)
    # returns: 3 x 3 skew symmetric matrix from *x*
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])

def reconstruct_one_point(pt1, pt2, m1, m2):
    """
        pt1 and m1 * X are parallel and cross product = 0
        pt1 x m1 * X  =  pt2 x m2 * X  =  0
    """
    A = np.vstack([
        np.dot(skew(pt1), m1),
        np.dot(skew(pt2), m2)
    ])
    # Compute the factor by Singular Value Decomposition
    U, S, V = np.linalg.svd(A)
    P = np.ravel(V[-1, :4]) 
    return P / P[3]


# Linear triangulation to find the 3D point X where p1 = m1 * X and p2 = m2 * X.
def linear_triangulation(p1, p2, m1, m2):
    """
    Solve AX = 0. parameters:: p1, p2: 2D points in homogeneous or catesian coordinates. Shape (3 x n) and m1, m2: Camera matrices associated with p1 and p2. Shape (3 x 4)
    """
    num_points = p1.shape[1]
    res = np.ones((4, num_points))

    for i in range(num_points):
        A = np.asarray([
            (p1[0, i] * m1[2, :] - m1[0, :]),
            (p1[1, i] * m1[2, :] - m1[1, :]),
            (p2[0, i] * m2[2, :] - m2[0, :]),
            (p2[1, i] * m2[2, :] - m2[1, :])
        ])
        # AX = 0: Factors of the scalar decomposition of A
        _, _, V = np.linalg.svd(A)
        X = V[-1, :4]
        res[:, i] = X / X[3]
    # returns 4 x n homogenous 3d triangulated points
    return res

# Computes the (right) epipole from a fundamental matrix F.
def compute_epipole(F):
    """ 
        (Use with F.T for left epipole.)
    """
    # return null space of F (Fx=0)
    U, S, V = np.linalg.svd(F)
    e = V[-1]
    return e / e[2]


def plot_epipolar_lines(p1, p2, F, show_epipole=False):
    """ Plot the points and epipolar lines. P1' F P2 = 0 """
    plt.figure()
    plt.suptitle('Epipolar lines', fontsize=16)

    plt.subplot(1, 2, 1, aspect='equal')
    # Plot the epipolar lines on img1 with points p2 from the right side
    # L1 = F * p2
    plot_epipolar_line(p1, p2, F, show_epipole)
    plt.subplot(1, 2, 2, aspect='equal')
    # Plot the epipolar lines on img2 with points p1 from the left side
    # L2 = F' * p1
    plot_epipolar_line(p2, p1, F.T, show_epipole) #transpose of F

# Plot the epipole and epipolar line F*x=0 in an image given the corresponding points.
def plot_epipolar_line(p1, p2, F, show_epipole=False):
    """ 
        Parameters: F is the fundamental matrix and p2 are the point in the other image.
    """
    lines = np.dot(F, p2)
    pad = np.ptp(p1, 1) * 0.01
    mins = np.min(p1, 1)
    maxes = np.max(p1, 1)

    # epipolar line parameter and values
    xpts = np.linspace(mins[0] - pad[0], maxes[0] + pad[0], 100)
    for line in lines.T:
        ypts = np.asarray([(line[2] + line[0] * p) / (-line[1]) for p in xpts])
        valid_idx = ((ypts >= mins[1] - pad[1]) & (ypts <= maxes[1] + pad[1]))
        plt.plot(xpts[valid_idx], ypts[valid_idx], linewidth=1)
        plt.plot(p1[0], p1[1], 'ro')

    if show_epipole:
        epipole = compute_epipole(F)
        plt.plot(epipole[0] / epipole[2], epipole[1] / epipole[2], 'r*')


# Compute camera matrix from pairs of 2D-3D correspondences in homogeneous coordinates.
def compute_P(p2d, p3d):
    n = p2d.shape[1]
    if p3d.shape[1] != n:
        raise ValueError('Number of points do not match.')

    # create matrix for DLT(Direct linear transformation) solution
    M = np.zeros((3 * n, 12 + n))
    for i in range(n):
        M[3 * i, 0:4] = p3d[:, i]
        M[3 * i + 1, 4:8] = p3d[:, i]
        M[3 * i + 2, 8:12] = p3d[:, i]
        M[3 * i:3 * i + 3, i + 12] = -p2d[:, i]
    U, S, V = np.linalg.svd(M)
    return V[-1, :12].reshape((3, 4))

# Compute the second camera matrix (assuming P1 = [I 0]) from a fundamental matrix.
def compute_P_from_fundamental(F):
    e = compute_epipole(F.T)  # left epipole
    Te = skew(e)
    return np.vstack((np.dot(Te, F.T).T, e)).T

#Compute the second camera matrix (assuming P1 = [I 0]) from an essential matrix. E = [t]R
def compute_P_from_essential(E):
    
    U, S, V = np.linalg.svd(E)

    # Ensure rotation matrix are right-handed with positive determinant
    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V

    # create 4 possible camera matrices 
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    P2s = [np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T]

    # returns list of 4 possible camera matrices.
    return P2s 


def correspondence_matrix(p1, p2):
    p1x, p1y = p1[:2]
    p2x, p2y = p2[:2]

    return np.array([
        p1x * p2x, p1x * p2y, p1x,
        p1y * p2x, p1y * p2y, p1y,
        p2x, p2y, np.ones(len(p1x))
    ]).T


# Compute the fundamental or essential matrix from corresponding points (x1, x2 3*n arrays) using the 8 point algorithm.
def compute_image_to_image_matrix(x1, x2, compute_essential=False):
    
    # Each row in the A matrix below is constructed as [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1]
    A = correspondence_matrix(x1, x2)
    # compute linear least square solution
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # constrain F. Make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    if compute_essential:
        S = [1, 1, 0] # Force rank 2 and equal eigenvalues
    F = np.dot(U, np.dot(np.diag(S), V))

    return F

# Scale and translate image points so that centroid of the points are at the origin and avg distance to the origin is equal to sqrt(2).
def scale_and_translate_points(points):
    """
    parameter:: points: array of homogenous point (3 x n)
    """
    x = points[0]
    y = points[1]
    center = points.mean(axis=1)  # mean of each row
    cx = x - center[0] # center the points
    cy = y - center[1]
    dist = np.sqrt(np.power(cx, 2) + np.power(cy, 2))
    scale = np.sqrt(2) / dist.mean()
    # normalized matrix
    norm3d = np.array([
        [scale, 0, -scale * center[0]],
        [0, scale, -scale * center[1]],
        [0, 0, 1]
    ])
    # returns array of same input shape and its normalization matrix
    return np.dot(norm3d, points), norm3d

# Computes the fundamental or essential matrix from corresponding points using the normalized 8 point algorithm.
def compute_normalized_image_to_image_matrix(p1, p2, compute_essential=False):
    """ 
    parameters p1, p2: corresponding points with shape 3 x n
    """
    n = p1.shape[1]
    if p2.shape[1] != n:
        raise ValueError('Number of points do not match.')

    # preprocess image coordinates
    p1n, T1 = scale_and_translate_points(p1)
    p2n, T2 = scale_and_translate_points(p2)

    # compute F or E with the coordinates
    F = compute_image_to_image_matrix(p1n, p2n, compute_essential)

    # reverse preprocessing of coordinates
    # We know that P1' E P2 = 0
    F = np.dot(T1.T, np.dot(F, T2))
    
    # returns fundamental or essential matrix with shape 3 x 3
    return F / F[2, 2]


def compute_fundamental_normalized(p1, p2):
    return compute_normalized_image_to_image_matrix(p1, p2)


def compute_essential_normalized(p1, p2):
    return compute_normalized_image_to_image_matrix(p1, p2, compute_essential=True)
