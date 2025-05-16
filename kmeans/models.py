"""
This is the class file you will have to fill in.
"""

from __future__ import annotations
import numpy as np

from kmeans.kmeans import kmeans
from kmeans.kmeans import assign_step
from collections import namedtuple


class KmeansClassifier(object):
    """
    K-Means Classifier via Iterative Improvement.

    Attributes
    ----------
    k : int, default=10
        The number of clusters to form as well as the number of centroids to
        generate.
    tol : float, default=1e-6
        Value specifying our convergence criterion. If the ratio of the
        distance each centroid moves to the previous position of the centroid
        is less than this value, then we declare convergence.
    max_iter : int, default=500
        The maximum number of times the algorithm can iterate trying to optimize the centroid values,
        the default value is set to 500 iterations.
    cluster_centers : np.ndarray
        A Numpy array where each element is one of the k cluster centers.
    """

    def __init__(
        self, n_clusters: int = 2, max_iter: int = 500, threshold: float = 1e-6
    ):
        """
        Initiate K-Means with some parameters

        Parameters
        ----------
        n_clusters : int, default=10
            The number of clusters to form as well as the number of centroids to
            generate.
        max_iter : int, default=500
            The maximum number of times the algorithm can iterate trying to optimize the centroid values,
            the default value is set to 500 iterations.
        threshold : float, default=1e-6
            Value specifying our convergence criterion. If the ratio of the
            distance each centroid moves to the previous position of the centroid
            is less than this value, then we declare convergence.
        """
        self.k = n_clusters
        self.tol = threshold
        self.max_iter = max_iter
        self.cluster_centers = np.array([])

    def train(self, X: np.ndarray) -> None:
        """
        Compute K-Means clustering on each class label and store your result in self.cluster_centers.

        Parameters
        ----------
        X : np.ndarray
            inputs of training data, a 2D Numpy array

        Returns
        -------
        None
        """
        # TODO: train using kmeans() and assign values to cluster centers
        self.cluster_centers = kmeans(X, self.k, self.max_iter, self.tol)

    def predict(self, X: np.ndarray, centroid_assignments: list[int]) -> np.ndarray:
        """
        Predicts the label of each sample in X based on the assigned centroid_assignments.

        Parameters
        ----------
        X : np.ndarray
            A dataset as a 2D Numpy array.
        centroid_assignments : list[int]
            A Python list of 10 digits (0-9) representing the interpretations of the digits of the plotted centroids.

        Returns
        -------
        np.ndarray
            A Numpy array of predicted labels.
        """
        # TODO: complete this step only after having plotted the centroids!
        centroid_idx = assign_step(X, self.cluster_centers).astype(int)
        return np.array(centroid_assignments)[centroid_idx]

    def accuracy(
        self,
        data: tuple[list[np.ndarray], list[np.ndarray]],
        centroid_assignments: list[int],
    ) -> float:
        """
        Compute accuracy of the model when applied to data.

        Parameters
        ----------
        data : tuple[list[np.ndarray], list[np.ndarray]]
            A namedtuple including inputs and labels.
        centroid_assignments : list[int]
            A python list of 10 digits (0-9) representing your interpretations of the digits of the plotted centroids from plot_Kmeans (in order from left ot right).

        Returns
        -------
        float
            A float number indicating accuracy.
        """
        pred = self.predict(data.inputs, centroid_assignments)
        return np.mean(pred == data.labels)
