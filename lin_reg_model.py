"""
This file contains the source for the LinearRegression class

Brown cs1420, Spring 2025
"""

import numpy as np


def squared_error(predictions: np.ndarray, Y: np.ndarray) -> float:
    """
    Computes L2 loss (sum squared loss) between true values, Y, and predictions.

    Parameters
    ----------
    predictions : np.ndarray
        A 1D Numpy array of the same size of Y.
    Y : np.ndarray
        A 1D Numpy array with real values (float64)

    Returns
    -------
    float
        L2 loss using predictions for Y.
    """
    # TODO: Add your solution code here.
    loss = 0
    Y = np.array(Y)
    shape = Y.shape[0]
    for i in range(shape):
        loss = loss + (Y[i] - predictions[i])**2
    return loss


class LinearRegression:
    """
    LinearRegression model that minimizes squared error using matrix inversion.
    """

    def __init__(self, n_features: int) -> None:
        """
        Parameters
        ----------
        n_features: int
            The number of features in the classification problem.

        Attributes
        ----------
        n_feature : int
            The number of features in the classification problem.
        weights : np.ndarray
            The weights of the linear regression model.
        """
        self.n_features = n_features + 1  # An extra feature added for the bias value
        self.weights = np.zeros(self.n_features)

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Trains the LinearRegression model weights using either
        stochastic gradient descent or matrix inversion.

        Parameters
        ----------
        X : np.ndarray
            2D Numpy array where each row contains an example, padded by 1 column for the bias.
        Y : np.ndarray
            1D Numpy array containing the corresponding values for each example.
        """
        self.train_solver(X, Y)

    def train_solver(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Trains the LinearRegression model by finding the optimal set of weights
        using matrix inversion.

        Parameters
        ----------
        X: np.ndarray
            2D Numpy array where each row contains an example, padded by 1 column for the bias.
        Y: np.ndarray
            1D Numpy array containing the corresponding values for each example.
        """
        # TODO: Add your solution code here.
        XTX = X.T @ X
        inv = np.linalg.pinv(XTX)
        self.weights = (inv @ X.T) @ Y


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns predictions of the model on a set of examples X.

        Parameters
        ----------
        X : np.ndarray
            A 2D Numpy array where each row contains an example, padded by 1 column for the bias.

        Returns
        -------
        np.ndarray
            A 1D Numpy array with one element for each row in X containing the predicted value.
        """
        # TODO: Add your solution code here.
        size = X.shape[0]
        predictions = np.zeros(size)
        for i in range(size):
            predictions[i] = np.dot(self.weights, X[i])
        return predictions

    def loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Returns the total squared error on some dataset (X, Y).

        Parameters
        ----------
        X : np.ndarray
            2D Numpy array where each row contains an example, padded by 1 column for the bias.
        Y : np.ndarray
            1D Numpy array containing the corresponding values for each example.

        Returns
        -------
        float
            A float number which is the squared error of the model on the dataset.
        """
        predictions = self.predict(X)
        return squared_error(predictions, Y)

    def average_loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Returns the mean squared error on some dataset (X, Y).

        MSE = Total squared error/# of examples

        Parameters
        ----------
        X : np.ndarray
            2D Numpy array where each row contains an example, padded by 1 column for the bias.
        Y : np.ndarray
            1D Numpy array containing the corresponding values for each example.

        Returns
        -------
        float
            A float number which is the mean squared error of the model on the dataset.
        """
        return self.loss(X, Y) / X.shape[0]