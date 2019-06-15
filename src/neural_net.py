import numpy as np

np.float_ = np.float64


class EzNet(object):

    def __init__(self, num_ft, h_layers=None, num_types=1):
        """
        Creates an EzNet with the specified number of features, specified hidden layer
        configuration, and specified number of output classification types
        :param num_ft: the number of features per training/input example, not including the bias
        unit
        :param h_layers: array of hidden layers where each element corresponds to a hidden layer
        with the specified number of units. For example, [2, 3] will create 2 hidden layers: the
        fist with 2 nodes, the 2nd with 3 units. Default is no hidden layers
        :param num_types: the number of output classification nodes. Default is 1, which is
        binary classification
        """
        self._means = np.zeros(shape=num_ft)
        self._stds = np.ones(shape=num_ft)
        self._thetas = []
        if h_layers is None:
            self._thetas.append(np.random.rand(num_types, num_ft + 1) - 0.5)
        else:
            self._thetas.append(np.random.rand(h_layers[0], num_ft + 1) - 0.5)
            for i in range(1, len(h_layers)):
                self._thetas.append(np.random.rand(h_layers[i], h_layers[i - 1] + 1) - 0.5)
            self._thetas.append(np.random.rand(num_types, h_layers[-1] + 1) - 0.5)

    def set_normalization_params(self, x):
        """
        Sets the normalization factors using the values in matrix x. The values of x are scaled
        column wise (i.e. there are normalization factors for each column of x)
        :param x: matrix for which to normalize
        :return: None
        """
        for i in range(0, np.size(x, axis=1)):
            col = x[:, i]
            self._means[i] = np.mean(col)
            self._stds[i] = np.std(col)

    def normalize(self, x):
        """
        Normalizes x by column. Each value v in x is scaled to (v - mean) / std, where mean and
        std are the mean and standard deviation for that column of x.
        :param x: the matrix of values to scale
        :return: a matrix of the normalized values of x
        """
        x_norm = np.zeros(shape=np.shape(x))
        for i in range(0, np.size(x, axis=1)):
            col = x[:, i]
            if self._stds[i] != 0:
                x_norm[:, i] = (col - self._means[i]) / self._stds[i]
            else:
                x_norm[:, i] = 0
        return x_norm

    def sigmoid(self, z):
        """
        Computes the sigmoid function of all values within matrix z. The sigmoid function is
        defined as 1 / (1 + e^-z)
        :param z: matrix of values for which to compute sigmoid. z is clipped to a value from
        -500 to 500 in order to prevent overflow in the sigmoid calculation
        :return: matrix of sigmoid function applied to each element of z, same shape as z
        """
        np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def d_sigmoid(self, z):
        """
        Computes the derivative of the sigmoid function. If we call sigmoid(z) s, then the
        derivative is simply s * (1 - s)
        :param z: the matrix for which to compute the derivative of sigmoid function
        :return: a matrix containing the derivative of the sigmoid for each value in z,
        same shape as z
        """
        sig = self.sigmoid(z)
        return sig * (1 - sig)

    def feed_forward(self, x):
        """
        Feeds forward x through the network and computes the most-likely classification of each
        example in x.
        :param x: the feature matrix to feed through the neural net. Each row of x should be a
        separate example and each column of x should be the value of a feature. Examples in x
        should not include the bias unit. x must have num_ft number of columns (refer to init)
        :return: a tuple containing an array with the predictions for each row of x, a list of
        the input matrices to each layer, and a list of the activation function results for each
        layer.
        """
        z = list()
        a = list()
        a.append(np.concatenate((np.ones(shape=(np.size(x, axis=0), 1)), x), axis=1).T)
        for theta in self._thetas:
            z.append(np.dot(theta, a[-1]))
            n_a = self.sigmoid(z[-1])
            if theta is not self._thetas[-1]:
                n_a = np.concatenate((np.ones(shape=(1, np.size(n_a, axis=1))), n_a), axis=0)
            a.append(n_a)
        return np.argmax(a[-1], axis=0), z, a

    def backpropogate(self, x, y, learn_rate, a_list, z_list):
        """
        Executes the backpropagation algorithm for each example in x, and appends the total error
        to j.
        :param x: the matrix of training values to perform backpropagation on. Each row of x
        should be a training example and each column of x should be a feature. x should not have
        bias units.
        :param y: the correct classifications for each training example in x. Each row of y
        should be the correct classification for the corresponding row of x
        :param learn_rate: the learning rate of the neural network
        :param a_list: the list of activation function results. List of arrays. Index of list
        corresponds to which layer (output values are last element of list)
        :param z_list: the list of summations from previous layers' activation functions. Index
        of list corresponds to which layer the summation is used in (last element of list is
        summation values used to calculate outputs)
        backpropagation is appended to err_list
        :return: None
        """
        delta_list = []
        grad_list = []
        m = np.size(x, axis=0)
        for i in range(0, m):
            y_vect = np.zeros(shape=(np.size(self._thetas[-1], axis=0), 1))
            if (np.size(y_vect, axis=0)) == 1:
                y_vect[0, 0] = y[i]
            else:
                y_vect[y[i], 0] = 1
            a_ind = len(a_list) - 2
            z_ind = len(z_list) - 1
            for j in range(len(self._thetas) - 1, -1, -1):
                if j == len(self._thetas) - 1:
                    a = a_list[-1][:, i].reshape(-1, 1)
                    z = z_list[-1][:, i].reshape(-1, 1)
                    delta = (a - y_vect) * self.d_sigmoid(z)
                    delta_list.insert(0, delta)
                    prev_a = a_list[a_ind][:, i].reshape(1, -1)
                    grad_list.insert(0, np.dot(delta, prev_a))
                else:
                    z = z_list[z_ind][:, i].reshape(-1, 1)
                    delta = np.dot(self._thetas[j + 1].T, delta_list[0])[1:, :] * self.d_sigmoid(z)
                    delta_list.insert(0, delta)
                    prev_a = a_list[a_ind][:, i].reshape(1, -1)
                    grad_list.insert(0, np.dot(delta, prev_a))
                a_ind -= 1
                z_ind -= 1
            for k in range(0, len(self._thetas)):
                self._thetas[k] -= learn_rate * grad_list[k]

    def compute_loss(self, x, y):
        """
        Computes the loss function for the given x and y.
        :param x: feature matrix. Each row is a training example, each row is a training example
        and each column is a feature
        :param y: the correct classifications of each corresponding training example in x
        :return: the loss function for x and y
        """
        preds, z_list, a_list = self.feed_forward(x)
        m = np.size(x, axis=0)
        loss = 0
        for i in range(0, m):
            y_vect = np.zeros(shape=(np.size(self._thetas[-1], axis=0), 1))
            y_vect[y[i], 0] = 1
            probs = np.reshape((a_list[-1][:, i]), np.shape(y_vect))
            loss += 1 / (2 * m) * np.sum((probs - y_vect) ** 2)
        return loss

    def train(self, x_train, y_train, learn_rate, epochs, x_test, y_test):
        """
        Trains the neural network using the specified dataset and learning rate for the specified
        number of epochs. Note that the network should be normalized before training.
        :param x_train: the matrix of training examples. Each row in x is a training example. Each
        column in x corresponds to a feature. x should not include bias units
        :param y_train: the correct classfications for each training example in x. Each row of y
        corresponds to the correct classification for the training example specified by the
        corresponding row of x
        :param learn_rate: learning rate for neural network
        :param epochs: the number of epochs to train the network. One epoch means that the
        network will train using the whole dataset once.
        :param x_test: a set of inputs to test the model on after each training epoch. Must have
        the same number of features as x_train
        :param y_test: the set of correct classifications (y values) for x_test
        :return: A tuple of 3 lists. The lists are the values of of the test error after each
        epoch, the training error after each epoch, and the loss function value after each epoch.
        """
        self.set_normalization_params(x_train)
        x_norm_train = self.normalize(x_train)
        acc_list_test_set = list()
        acc_list_train_set = list()
        loss_func_vals = list()
        for i in range(0, epochs):
            preds, z_list, a_list = self.feed_forward(x_norm_train)
            self.backpropogate(x_norm_train, y_train, learn_rate, a_list, z_list)

            pred_train = self.feed_forward(x_norm_train)[0]
            pred_train = np.reshape(pred_train, np.shape(y_train))
            acc_train = np.sum((pred_train == y_train)) / np.size(y_train, 0)
            acc_list_train_set.append(acc_train)

            test_pred = self.predict(x_test)[0]
            predictions = np.reshape(test_pred, np.shape(y_test))
            acc = np.sum((predictions == y_test)) / np.size(y_test, 0)
            acc_list_test_set.append(acc)

            loss_func_vals.append(self.compute_loss(x_train, y_train))
        return acc_list_test_set, acc_list_train_set, loss_func_vals

    def predict(self, x):
        """
        Computes the prediction for x and returns it. The predictions are stored in an array
        where each value corresponds to the prediction for its respective feature vector. The
        neural net should be normalized prior to making predictions.
        :param x: a matrix of samples to make predictions for. Each row of x corresponds to one
        sample and each column of x is a feature. x should not include bias units
        :return: a tuple containing an array with the predictions for each row of x, a list of
        the input matrices to each layer, and a list of the final values for each
        layer.
        """
        x_norm_pred = self.normalize(x)
        return self.feed_forward(x_norm_pred)
