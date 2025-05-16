"""
This file implements a Logistic Regression classifier

Brown cs1420, Spring 2025
"""

import random

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def softmax(x: np.ndarray) -> np.ndarray:
    """Calculates element-wise softmax of the input array

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    np.ndarray
        Softmax output of the given array x
    """
    e = np.exp(x - np.max(x))
    return (e + 1e-6) / (np.sum(e) + 1e-6)


class LogisticRegression:
    """
    Multiclass logistic regression model that learns weights using
    stochastic gradient descent (SGD).
    """

    def __init__(
        self, n_features: int, n_classes: int, batch_size: int, conv_threshold: float
    ) -> None:
        """Constructor for a LogisticRegression classifier instance

        Parameters
        ----------
        n_features : int
            The number of features in the classification problem
        n_classes : int
            The number of classes in the classification problem
        batch_size : int
            Batch size to use in SGD
        conv_threshold : float
            Convergence threshold; once reached, discontinues the optimization loop

        Attributes
        ----------
        alpha : int
            The learning rate used in SGD
        weights : np.ndarray
            Model weights
        """
        self.n_classes = n_classes
        self.n_features = n_features
        self.weights = np.zeros(
            (n_classes, n_features + 1)
        )  # NOTE: An extra row added for the bias
        self.alpha = 0.03
        self.batch_size = batch_size
        self.conv_threshold = conv_threshold

    def train(self, X: np.ndarray, Y: np.ndarray) -> int:
        """This implements the main training loop for the model, optimized
        using stochastic gradient descent.

        Parameters
        ----------
        X : np.ndarray
            A 2D Numpy array containing the datasets. Each row corresponds to one example, and
            each column corresponds to one feature. Padded by 1 column for the bias term.
        Y : np.ndarray
            A 1D Numpy array containing the labels corresponding to each example.

        Returns
        -------
        int
            Number of epochs taken to converge
        """
        # TODO: Add your solution code here.
        converged = False
        epochs = 0
        n = len(X)
        l_0 = np.inf


        while converged is False:
            epochs += 1

            shuffle = np.random.permutation(n)
            X = X[shuffle]
            Y = Y[shuffle]

            for i in range(int(np.ceil(n/self.batch_size))):
                new_X = X[i*self.batch_size:(i+1)*self.batch_size]
                new_Y = Y[i*self.batch_size:(i+1)*self.batch_size]

                new_n = len(new_X)
                l_weights = np.zeros(self.weights.shape)
                for (x, y) in zip(new_X, new_Y):
                    for j in range(self.n_classes):
                        sm = softmax(self.weights @ x)
                        if y == j:
                            l_weights[j] += (sm[j]-1) * x
                        else:
                            l_weights[j] += sm[j] * x
                self.weights -= (self.alpha/new_n)*l_weights
            loss = self.loss(X, Y)
            if np.abs(loss - l_0) < self.conv_threshold:
                converged = True
            else:
                l_0 = loss
        return epochs

    def loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Calculates average log loss on the predictions made by the model
        on dataset X against the corresponding labels Y.

        Parameters
        ----------
        X : np.ndarray
            2D Numpy array representing a dataset. Each row corresponds to one example,
            and each column corresponds to one feature. Padded by 1 column for the bias.
        Y : np.ndarray
            1D Numpy array containing the corresponding labels to each example in dataset X.

        Returns
        -------
        float
            Average loss of the model on the dataset
        """
        # TODO: Add your solution code here.
        loss = 0
        for (x, y) in zip(X, Y):
            loss -= np.log(softmax(self.weights @ x)[y])
            # print(y)
            # print(np.log(softmax(self.weights @ x)))
            # for j in range(self.n_classes):
            #     if y == j:
            #         loss -= np.log(softmax(self.weights @ x))
        return -np.average(loss)
    


    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute predictions based on the learned parameters and examples X

        Parameters
        ----------
        X : np.ndarray
            A 2D Numpy array representing a dataset. Each row corresponds to one example,
            and each column corresponds to one feature. Padded by 1 column for the bias.

        Returns
        -------
        np.ndarray
            1D Numpy array of predictions corresponding to each example in X
        """
        # TODO: Add your solution code here.
        predictions = self.weights @ X.T
        return np.argmax(predictions, axis=0)
    

    def accuracy(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Outputs the accuracy of the trained model on a given test
        dataset X and labels Y.

        Parameters
        ----------
        X : np.ndarray
            A 2D Numpy array representing a dataset. Each row corresponds to one example,
            and each column corresponds to one feature. Padded by 1 column for the bias.
        Y : np.ndarray
            1D Numpy array containing the corresponding labels to each example in dataset X.

        Returns
        -------
        float
            Accuracy percentage (between 0 and 1) on the given test set.
        """
        # TODO: Add your solution code here.
        n = X.shape[0]
        predictions = self.predict(X)
        # correct = 0
        # for i in range(n):
        #     if predictions[i] == Y.T[i]:
        #         correct += 1

        correct = np.count_nonzero(predictions==Y.T)
        return correct / n
    
    def confusion(self, X: np.ndarray, Y: np.ndarray):
        predictions = self.predict(X)
        print(predictions)
        cm = confusion_matrix(Y, predictions)
        plt.figure()
        sns.heatmap(cm, annot=True)
        plt.xlabel("Predictions")
        plt.ylabel("Actual")
        plt.title("Linear Regression Confusion Matrix")
        plt.show()


