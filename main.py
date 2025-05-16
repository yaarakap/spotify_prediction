import pandas as pd
import numpy as np
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from lin_reg_model import LinearRegression
# from log_reg_model import LogisticRegression
from reg_log_reg_model import RegularizedLogisticRegression

from kmeans.models import KmeansClassifier
from collections import namedtuple


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression 

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier



def get_and_split_data():
    data = pd.read_csv("/Users/yaarakaplan/Classes/ML/capstone/data/data (1).csv")

    data = data.drop(columns=["song_title","artist"])


    train, test = train_test_split(data, test_size=0.2) # 20% test data

    atts = ["acousticness", "energy", "danceability","duration_ms","instrumentalness", "speechiness","valence"]
    x_train = np.array(train[atts])
    y_train = np.array(train["target"])

    x_test = np.array(test[atts])
    y_test = np.array(test["target"])
    return x_train, x_test, y_train, y_test


def linreg(X_train, X_test, Y_train, Y_test) -> None:
    """
    Helper function that tests LinearRegression.
    """
    num_features = X_train.shape[1]

    # Padding the inputs with a bias
    X_train_b = np.append(X_train, np.ones((len(X_train), 1)), axis=1)
    X_test_b = np.append(X_test, np.ones((len(X_test), 1)), axis=1)

    print(f"Running models on dataset")

    #### Matrix Inversion ######
    print("---- LINEAR REGRESSION w/ Matrix Inversion ---")
    solver_model = LinearRegression(num_features)
    solver_model.train(X_train_b, Y_train)
    print(f"Average Training Loss: {solver_model.average_loss(X_train_b, Y_train)}")
    print(f"Average Testing Loss: {solver_model.average_loss(X_test_b, Y_test)}")


def logreg(X_train, X_test, Y_train, Y_test) -> float:
    """Runs the model training and test loop on the census dataset.

    Returns
    -------
    float
        Returns model accuracy
    """
    num_features = X_train.shape[1]
    print(f"num features: {num_features}")

    # Add a bias
    X_train_b = np.append(X_train, np.ones((len(X_train), 1)), axis=1)
    X_test_b = np.append(X_test, np.ones((len(X_test), 1)), axis=1)

    ### Logistic Regression ###
    model = LogisticRegression(num_features, 2, 50, 0.00001)
    num_epochs = model.train(X_train_b, Y_train)
    acc = model.accuracy(X_test_b, Y_test) * 100
    print(f"Test Accuracy: {acc}%")
    print(f"Number of Epochs: {num_epochs}")

    model.confusion(X_test_b, Y_test)

    return acc

def regularized_log_reg():
    """
    Main driving function: trains model using regularized logistic regression and plots
    results against different lambda values.
    """
    X_train, X_val, Y_train, Y_val = get_and_split_data()
    X_train = np.append(X_train, np.ones((len(X_train), 1)), axis=1)
    X_val = np.append(X_val, np.ones((len(X_val), 1)), axis=1)

    X_train_val = np.concatenate((X_train, X_val))
    Y_train_val = np.concatenate((Y_train, Y_val))

    RR = RegularizedLogisticRegression()
    RR.train(X_train, Y_train)
    print(f"Train Accuracy: {RR.accuracy(X_train, Y_train)}")
    print(f"Validation Accuracy: {RR.accuracy(X_val, Y_val)}")

    print(RR.predict(X_val))
    RR.confusion(X_val, Y_val)


def runKMeans() -> None:
    """
    Trains, plots, and tests K-Means classifier on digits.csv dataset.

    Returns
    -------
    None
    """
    NUM_CLUSTERS = 2  # DO NOT CHANGE
    random.seed(1)
    np.random.seed(1)

    Dataset = namedtuple("Dataset", ["inputs", "labels"])

    train_inputs, test_inputs, train_labels, test_labels = get_and_split_data()

    # all_data = Dataset(inputs, labels)
    train_data = Dataset(train_inputs, train_labels)
    test_data = Dataset(test_inputs, test_labels)
    print("Shape of training data inputs: ", train_data.inputs.shape)
    print("Shape of test data inputs:", test_data.inputs.shape)

    # Train K-Means Classifier
    kmeans_model = KmeansClassifier(NUM_CLUSTERS)
    kmeans_model.train(train_data.inputs)
    cm = confusion_matrix(test_labels, kmeans_model.predict(test_inputs, [1, 0]))
    plt.figure()
    sns.heatmap(cm, annot=True)
    plt.xlabel("Predictions")
    plt.ylabel("Actual")
    plt.title("My KMeans Confusion Matrix")
    plt.show()


def kmeans():
    # prep data
    data = pd.read_csv("/Users/yaarakaplan/Classes/ML/capstone/data/data (1).csv")

    data = data.drop(columns=["song_title","artist"])

    train, test = train_test_split(data, test_size=0.2) # 20% test data

    atts = ["acousticness", "energy", "danceability","duration_ms","instrumentalness", "speechiness","valence", "target"]
    train = train[atts]
    test = test[atts]

    kmeans = KMeans(n_clusters=2).fit(train.drop(columns=["target"])) # using all data
    train["kmeans"] = kmeans.labels_

    # plotting clusters
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = train["danceability"]
    y = train["duration_ms"]
    z = train["target"]

    ax.scatter(x,y,z,c=train['kmeans'])
    ax.set_xlabel("Danceability")
    ax.set_ylabel("Duration (ms)")
    ax.set_zlabel("Target")
    ax.set_title("3D Scatter Plot of Songs Clustered")
    plt.show()

    print(kmeans.predict(test.drop(columns=["target"])))
    cm = confusion_matrix(test["target"], kmeans.predict(test.drop(columns=["target"])))
    plt.figure()
    sns.heatmap(cm, annot=True)
    plt.xlabel("Predictions")
    plt.ylabel("Actual")
    plt.title("Sklearn K-Means Confusion Matrix")
    plt.show()


def knn(X_train, X_test, Y_train, Y_test):
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    print(knn.predict(X_test))
    print(knn.score(X_test, Y_test))

    cm = confusion_matrix(Y_test, knn.predict(X_test))
    plt.figure()
    sns.heatmap(cm, annot=True)
    plt.xlabel("Predictions")
    plt.ylabel("Actual")
    plt.title("KNN Confusion Matrix")
    plt.show()



def main():
    """
    Main driving function.
    """
    X_train, X_test, Y_train, Y_test = get_and_split_data()
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    
    # # # linear regression
    # # print("linear regression: ")
    # # random.seed(0)
    # # np.random.seed(0)
    # # linreg(X_train, X_test, Y_train, Y_test)

    # logistic regression
    print("sklearn logistic regression")
    lr = LogisticRegression(solver='lbfgs', max_iter=400).fit(X_train, Y_train)
    lr_scores = cross_val_score(lr, X_train, Y_train, cv=10, scoring="f1")
    print(np.mean(lr_scores))
    print(lr.predict(X_test))

    cm = confusion_matrix(Y_test, lr.predict(X_test))
    plt.figure()
    sns.heatmap(cm, annot=True)
    plt.xlabel("Predictions")
    plt.ylabel("Actual")
    plt.title("Logistic Regression Confusion Matrix")
    plt.show()

    # commented out code bc it does not work :(
    # print("normal logistic regression: ")
    # np.random.seed(0)
    # logreg(X_train, X_test, Y_train, Y_test) # why is this giving bad accuracy?
    # # predicting all positive???

    # print("regularized logistic regression: ")
    # np.random.seed(16)
    # regularized_log_reg()

    print("kmeans clustering")
    runKMeans()
    kmeans()

    print("k nearest neighbors")
    knn(X_train, X_test, Y_train, Y_test)



if __name__ == "__main__":
    main()
