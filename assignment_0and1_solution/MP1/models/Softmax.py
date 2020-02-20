import numpy as np
import matplotlib.pyplot as plt
import math


class Softmax():
    def __init__(self):
        """
        Initialises Softmax classifier with initializing
        weights, alpha(learning rate), number of epochs
        and regularization constant.
        """
        self.w = None
        self.alpha = 1e-1  # learning rate .01
        self.epochs = 40
        self.reg_const = 1e-4  # regularization constant

    def plot_graph(self, errors):
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.plot(range(len(errors)), errors, c='green')
        plt.ylabel('Number of Misclassifications')
        plt.xlabel('Number of Epochs')
        plt.title('Errors vs Epochs')
        plt.show()

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

    def calc_gradient(self, X_train, y_train):
        """
        Calculate gradient of the softmax loss

        Inputs have dimension D, there are C classes, and we operate on minibatches
        of N examples.

        Inputs:
        - X_train: A numpy array of shape (N, D) containing a minibatch of data.
        - y_train: A numpy array of shape (N,) containing training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.

        Returns:
        - gradient with respect to weights W; an array of same shape as W
        """
        num_classes = 10
        num_train, D = X_train.shape
        # initializing gradient of weights with 0
        grad_w = np.zeros(self.w.shape)
        for i in range(num_train):  # num_train
            # computing scores for each X for 10 classes
            scores = np.dot(X_train[i], self.w)
            logK = np.max(scores)  # max x_i.W
            scores -= logK
            probabilities = np.exp(scores) / np.sum(np.exp(scores))
            # print(np.sum(probabilities))
            probabilities[y_train[i]] -= 1
            for c in range(num_classes):
                grad_w[:, c] += probabilities[c] * X_train[i, :]

        grad_w /= num_train
        grad_w += self.reg_const * self.w

        return grad_w

    def train(self, X_train, y_train):
        """
        Train Softmax classifier using stochastic gradient descent.

        Inputs:
        - X_train: A numpy array of shape (N, D) containing training data;
        N examples with D dimensions
        - y_train: A numpy array of shape (N,) containing training labels;

        Hint : Operate with Minibatches of the data for SGD
        """
        num_train, D = X_train.shape  # num_train = N
        num_class = 10
        self.w = np.random.uniform(-1, 1, size=(D, num_class))
        # print(self.w.shape)
        errors = []
        epoch = 0
        while epoch < self.epochs:

            scores = np.matmul(X_train, self.w)
            # finding row-wise maximum score
            predictions = scores.argmax(axis=1)
            # np.sum(array) counts true value in the array
            error = num_train - np.sum(predictions == y_train)
            errors.append(error)
            self.alpha = self.exp_decay(epoch)
            print('epoch:', epoch, ', error:', error, ' alpha: ', self.alpha)
            epoch += 1

            dw = self.calc_gradient(X_train, y_train)
            self.w = self.w - self.alpha * dw

        self.plot_graph(errors)

    def predict(self, X_test):
        """
        Use the trained weights of softmax classifier to predict labels for
        data points.

        Inputs:
        - X_test: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - pred: Predicted labels for the data in X_test. pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        scores = np.matmul(X_test, self.w)
        pred = scores.argmax(axis=1)
        return pred
