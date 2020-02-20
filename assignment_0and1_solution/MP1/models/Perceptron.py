import numpy as np
import scipy
import math
import matplotlib.pyplot as plt


class Perceptron():
    def __init__(self):
        """
        Initialises Perceptron classifier with initializing
        weights, alpha(learning rate) and number of epochs.
        alpha=1.0, epochs=20
        """
        self.w = None
        self.alpha = 1.0
        self.epochs = 50  # 50

    def plot_graph(self, errors):
        """
        This function plots errors vs epochs
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.plot(range(len(errors)), errors, c='green')
        plt.ylabel('Number of Misclassifications')
        plt.xlabel('Number of Epochs')
        plt.title('Errors vs Epochs')
        plt.show()

    def step_decay(self, epoch):
        """
        This function implements step decay of the learning rate using
        following equation
            lr = lr0 * drop^floor(t / epochs_drop)
            lr0 = initial learning rate
            lr = new learning rate
            t = epoch
            drop = learning rate dropping rate
            epochs_drop = after how many epochs learning rate will be dropped
        """
        lr0 = 3
        drop = 0.5
        epochs_drop = 5.0
        lr = lr0 * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lr

    def exp_decay(self, epoch):
        """
        This function implements the following equation for implementing
        exponential decay of the learning rate.
            lr = lr0 * e^(−kt)
            where
                lr0 = initial learning rate
                k = hyperparameter
                t = epoch
                lr = new learning rate
        """
        lr0 = 1e-1
        k = 0.3
        lr = lr0 * math.exp(-k * epoch)
        return lr

    def train(self, X_train, y_train):
        """
        Train the Perceptron classifier. Use the perceptron update rule
        as introduced in Lecture 3.

        Inputs:
        - X_train: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y_train: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        print(X_train.shape)
        print(y_train.shape)
        num_train, D = X_train.shape
        num_class = 10
        errors = []
        # initializing weights with uniform distribution from -1 to 1
        self.w = np.random.uniform(-1, 1, size=(D, num_class))
        # initializing bias, I did not update bias. I used the bias to calculate the errors for each iteration
        bias = np.random.uniform(-1, 1, size=(num_train, num_class))

        epoch = 0
        while epoch < self.epochs:
            # computing scores for each train example
            scores = np.matmul(X_train, self.w) + bias
            # finding row-wise maximum score
            predictions = scores.argmax(axis=1)
            error = num_train - np.sum(predictions == y_train)
            errors.append(error)
            # self.alpha = self.exp_decay(epoch)
            self.alpha = self.step_decay(epoch)
            print('epoch:', epoch, ', error:', error, ' alpha: ', self.alpha)
            epoch += 1

            # updating weights
            for i in range(num_train):  # num_train
                for c in range(num_class):  # num_class
                    # W_c.X_i > W_y_i.X_i, from lecture the indicator function
                    if scores[i][c] > scores[i][y_train[i]]:
                        if c != y_train[i]:  # c != y_i
                            features = X_train[i]
                            # updating W_c
                            self.w[:, c] = self.w[:, c] - \
                                self.alpha * features.transpose()
                        else:
                            # updating W_y_i
                            self.w[:, c] = self.w[:, c] + \
                                self.alpha * features.transpose()

        self.plot_graph(errors)

    def predict(self, X_test):
        """
        Predict labels for test data using the trained weights.

        Inputs:
        - X_test: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.

        Returns:
        - pred: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        scores = np.matmul(X_test, self.w)
        pred = scores.argmax(axis=1)
        return pred
