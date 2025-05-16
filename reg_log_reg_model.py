import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def sigmoid_function(x: np.ndarray) -> np.ndarray:
    """
    Applies the sigmoid function element-wise.

    Parameters
    ----------
    x : np.ndarray
        An array of values.

    Returns
    -------
    np.ndarray
        An array containing the result of applying the sigmoid function to each element of x.
    """
    return 1.0 / (1.0 + np.exp(-x))


class RegularizedLogisticRegression:
    """
    Implement regularized logistic regression for binary classification.

    The weight vector w should be learned by minimizing the regularized loss
    \l(h, (x,y)) = log(1 + exp(-y <w, x>)) + \lambda \|w\|_2^2. In other words, the objective
    function that we are trying to minimize is the log loss for binary logistic regression
    plus Tikhonov regularization with a coefficient of \lambda.
    """

    def __init__(self) -> None:
        """
        Attributes
        ----------
        learningRate : float
            Learning rate (alpha).
        num_epochs : int
            Number of epochs to train for.
        batch_size : int
            Batch size in training.
        weights : np.ndarray
            The weights of the model.
        lmbda : float
            The Tikhanov Regularization constant.
        """
        self.learningRate = 0.00001  # Feel free to play around with this if you'd like, though this value will do
        self.num_epochs = 10000  # Feel free to play around with this if you'd like, though this value will do
        self.batch_size = 15  # Feel free to play around with this if you'd like, though this value will do
        self.weights = None

        #####################################################################
        #                                                                    #
        #    MAKE SURE TO SET THIS TO THE OPTIMAL LAMBDA BEFORE SUBMITTING    #
        #                                                                    #
        #####################################################################

        self.lmbda = 75  # TODO: tune this parameter

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Train the model, using batch stochastic gradient descent.
        Remember to use regularization when updating wegihts.

        Parameters
        ----------
        X : np.ndarray
            A 2D Numpy array where each row contains an example, padded by 1 column for the bias.
        Y : np.ndarray
            A 1D Numpy array containing the corresponding labels for each example.
        """
        # TODO: Add your solution code here.

        np.random.seed(16)

        n = len(X)
        self.weights = np.zeros(X.shape[1])

        for e in range(self.num_epochs):

            shuffle = np.random.permutation(n)
            X = X[shuffle]
            Y = Y[shuffle]

            for i in range(int(np.ceil(n/self.batch_size))):
                new_X = X[i*self.batch_size:(i+1)*self.batch_size]
                new_Y = Y[i*self.batch_size:(i+1)*self.batch_size]

                sig = sigmoid_function(np.dot(new_X, self.weights))
                gradient = np.dot(sig - new_Y, new_X)
                self.weights -= self.learningRate * (gradient/self.batch_size + 2*self.lmbda*self.weights)

                # loss = np.average(new_Y * np.log(sig) + (np.ones(new_Y.shape)-new_Y) * np.log(1-sig))
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predictions based on the learned parameters and examples X.

        Parameters
        ----------
        X : np.ndarray
            A 2D Numpy array where each row contains an example, padded by 1 column for the bias.

        Returns
        -------
        np.ndarray
            A 1D Numpy array with a 0 or 1 for each row in X, representing the predicted class.
        """
        # TODO: Add your solution code here.

        return np.where(sigmoid_function(np.dot(X, self.weights)) >= 0.5, 1, 0)


    def accuracy(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Output the accuracy of the trained model on a given testing dataset X and labels Y.

        Parameters
        ----------
        X : np.ndarray
            A 2D Numpy array where each row contains an example, padded by 1 column for the bias.
        Y : np.ndarray
            A 1D Numpy array containing the corresponding labels for each example.

        Returns
        -------
        float
            A float number indicating accuracy (between 0 and 1).
        """
        # TODO: Add your solution code here.
        n = X.shape[0]
        predictions = self.predict(X)
        correct = np.count_nonzero(predictions==Y.T)
        return correct / n

    def runTrainValSplit(
        self,
        lambda_list: list[float],
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
    ) -> tuple[list[float], list[float]]:
        """
        Given the training and validation data, fit the model with training data and validate it with
        respect to each lambda. Record the training error and validation error, which are equivalent
        to (1 - accuracy).

        Parameters
        ----------
        lambda_list : list[float]
            A list of lambdas.
        X_train : np.ndarray
            A 2D Numpy array for training where each row contains an example, padded by 1 column for the bias.
        Y_train : np.ndarray
            A 1D Numpy array for training containing the corresponding labels for each example.
        X_val : np.ndarray
            A 2D Numpy array for validation where each row contains an example, padded by 1 column for the bias.
        Y_val : np.ndarrray
            A 1D Numpy array for validation containing the corresponding labels for each example.

        Returns
        -------
        tuple[list[float], list[float]]
            A tuple (train_errors, val_errors), where
            - train_errors: a list of training errors with respect to the lambda_list
            - val_errors: a list of validation errors with respect to the lambda_list
        """
        train_errors = []
        val_errors = []
        # TODO: train model and calculate train and validation errors here for each lambda

        for lam in lambda_list:
            
            self.lmbda = lam
            self.train(X_train, Y_train)

            train_errors.append(1-self.accuracy(X_train, Y_train))
            val_errors.append(1-self.accuracy(X_val, Y_val))

        return train_errors, val_errors

    def _kFoldSplitIndices(self, dataset: np.ndarray, k: int) -> list[list[int]]:
        """
        Helper function for k-fold cross validation. Evenly split the indices of a
        dataset into k groups.

        For example, indices = [0, 1, 2, 3] with k = 2 may have an output
        indices_split = [[1, 3], [2, 0]].

        Please don't change this.

        Parameters
        ----------
        dataset : np.ndarray
            A Numpy array where each row contains an example.
        k : np.ndarray
            An integer, which is the number of folds.

        Returns
        -------
        list[list[int]]
            A list containing k lists of indices.
        """
        num_data = dataset.shape[0]
        fold_size = int(num_data / k)
        indices = np.random.permutation(num_data)
        indices_split = np.split(indices[: fold_size * k], k)
        return indices_split

    def runKFold(
        self, lambda_list: list[float], X: np.ndarray, Y: np.ndarray, k: int = 3
    ) -> list[float]:
        """
        Run k-fold cross validation on X and Y with respect to each lambda. Return all k-fold
        errors.

        Each run of k-fold involves k iterations. For an arbitrary iteration i, the i-th fold is
        used as testing data while the remaining k-1 folds are combined as one set of training data. The k results are
        averaged as the cross validation error.

        Parameters
        ----------
        lambda_list : list[float]
            A list of lambdas
        X : np.ndarray
            A 2D Numpy array where each row contains an example, padded by 1 column for the bias.
        Y : np.ndarray
            A 1D Numpy array containing the corresponding labels for each example.
        k : int
            The number of folds; k is 3 by default.

        Returns
        -------
        list[float]
            A list of k-fold errors with respect to the lambda_list.
        """
        initial_lmbda = self.lmbda
        k_fold_errors = []
        for lmbda in lambda_list:
            self.lmbda = lmbda
            # TODO: call _kFoldSplitIndices to split indices into k groups randomly
            groups = self._kFoldSplitIndices(X, k)
            # TODO: for each iteration i = 1...k, train the model using lmbda
            # on k-1 folds of data. Then test with the i-th fold.
            errs = []
            for i in range(k):

                X_test = X[groups[i]]
                Y_test = Y[groups[i]]

                new_lst = groups.copy()
                new_lst.pop(i)
                idx = [elt for group in new_lst for elt in group]
                idx = np.array(idx)

                X_train = X[idx]
                Y_train = Y[idx]

                self.train(X_train, Y_train)
                errs.append(1-self.accuracy(X_test, Y_test))

            # TODO: calculate and record the cross validation error by averaging total errors
            k_fold_errors.append(np.average(errs))
        self.lmbda = initial_lmbda  # reset lambda value to what it was before calling
        return k_fold_errors

    def plotError(
        self,
        lambda_list: list[float],
        train_errors: list[float],
        val_errors: list[float],
        k_fold_errors: list[float],
    ) -> None:
        """
        Produce a plot of the cost function on the training and validation sets, and the
        cost function of k-fold with respect to the regularization parameter lambda. Use this plot
        to determine a valid lambda.

        Parameters
        ----------
        lambda_list : list[float]
            A list of lambdas.
        train_errors : list[float]
            A list of training errors with respect to the lambda_list.
        val_errors : list[float]
            A list of validation errors with respect to the lambda_list.
        k_fold_errors : list[float]
            A list of k-fold errors with respect to the lambda_list.
        """
        plt.figure()
        plt.semilogx(lambda_list, train_errors, label="training error")
        plt.semilogx(lambda_list, val_errors, label="validation error")
        plt.semilogx(lambda_list, k_fold_errors, label="k-fold error")
        plt.xlabel("lambda")
        plt.ylabel("error")
        plt.legend()
        plt.show()

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

