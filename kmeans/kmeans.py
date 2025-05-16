import numpy as np
from random import sample


def init_centroids(k: int, inputs: np.ndarray) -> np.ndarray:
    """
    Selects k random rows from inputs and returns them as the chosen centroids.
    Hint: use random.sample

    Parameters
    ----------
    k : int
        Number of cluster centroids.
    inputs : np.ndarray
        A 2D Numpy array, each row of which is one input.

    Returns
    -------
    np.ndarray
        A Numpy array of k cluster centroids, one per row.
    """
    # TODO: Add your solution code here.
    return np.array(sample(list(inputs), k))


def assign_step(inputs: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Determines a centroid index for every row of the inputs using Euclidean Distance.

    Parameters
    ----------
    inputs : np.ndarray
        Inputs of data, a 2D Numpy array.
    centroids : np.ndarray
        A Numpy array of k current centroids.

    Returns
    -------
    np.ndarray
        A Numpy array of centroid indices, one for each row of the inputs.
    """
    # TODO: Add your solution code here.
    centroid_idx = np.zeros(inputs.shape[0])
    for i in range(inputs.shape[0]):
        norm = np.zeros(centroids.shape[0])
        for j in range(centroids.shape[0]):
            norm[j] = np.linalg.norm(inputs[i] - centroids[j])**2
        centroid_idx[i] = np.argmin(norm)
    return centroid_idx


def update_step(inputs: np.ndarray, indices: np.ndarray, k: int) -> np.ndarray:
    """
    Computes the centroid for each cluster.

    Parameters
    ----------
    inputs : np.ndarray
        Inputs of data, a 2D Numpy array.
    indices : np.ndarray
        A Numpy array of centroid indices, one for each row of the inputs.
    k : int
        Number of cluster centroids.

    Returns
    -------
    np.ndarray
        A Numpy array of k cluster centroids, one per row.
    """
    # TODO: Add your solution code here.
    centroids = np.zeros((k, inputs.shape[1]))
    for i in range(k):
        centroids[i] = np.average(inputs[indices==i], axis=0)
    return centroids


def kmeans(inputs: np.ndarray, k: int, max_iter: int, tol: float) -> np.ndarray:
    """
    Runs the K-means algorithm on n rows of inputs using k clusters via iterative improvement.

    Parameters
    ----------
    inputs : np.ndarray
        Inputs of data, a 2D Numpy array.
    k : int
        Number of cluster centroids.
    max_iter : int
        The maximum number of times the algorithm can iterate trying to optimize the centroid values.
    tol : float
        The tolerance we determine convergence with when compared to the ratio as stated on handout.

    Returns
    -------
    np.ndarray
        A Numpy array of k cluster centroids, one per row.
    """
    # TODO: Add your solution code here.
    centroids = init_centroids(k, inputs)
    for i in range(max_iter):
        prev_centroids = centroids.copy()
        centroid_idx = assign_step(inputs, centroids)
        centroids = update_step(inputs, centroid_idx, k)
        if np.all(np.linalg.norm(centroids-prev_centroids) < tol):
            break
    return centroids
    
