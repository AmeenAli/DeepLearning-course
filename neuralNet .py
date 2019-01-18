import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston


def sigmoid(x):
    """Return sigmoid function and its gradient on point x element-wise"""
    sig_val = 1. / (1. + np.exp(-x))

    return sig_val, np.multiply(sig_val, 1 - sig_val)


def squared_loss(z, y):
    """Returns the value of the squared loss on y and its gradient on z """
    return (z - y) ** 2, 2.*(z - y)


def norm_2_regularization(w, reg_factor):
    """Returns the regularization term and its gradient with respect to w"""
    return (reg_factor/2.) * np.linalg.norm(w) ** 2, reg_factor * w


class NeuralNetwork(object):
    class GradientError(Exception):
        pass

    def __init__(self, layer_layout, cost_function, activation_function, regularization_function=None,
                 regularization_factor=0., eta='constant', eta_init=0.01, zero_weights=False, check_gradients=False):
        # number of units (not counting the bias unit) in each layer
        # (There is one unit in each layer that always outputs 1 which
        # is called the bias unit)
        self._layer_layout = layer_layout
        self._cost_function = cost_function
        self._activation_function = activation_function
        if regularization_function is None:
            self._regularization_function = lambda w: (0, 0)
        else:
            self._regularization_function = lambda w: regularization_function(w, regularization_factor)
        self._eta_for_epoch = NeuralNetwork._eta_generator(eta, eta_init)
        self._num_layers = len(layer_layout)
        self._weights = self._init_weights(zero_weights)
        self._check_gradients = check_gradients
        self._epoch = 1

    def _init_weights(self, zero_weights):
        weights_per_layer = []
        for idx in xrange(1, self._num_layers):
            # add constant input(=bias)
            num_input_units_to_layer = self._layer_layout[idx - 1] + 1
            num_units_in_layer = self._layer_layout[idx]

            weight_shape = (num_input_units_to_layer, num_units_in_layer)
            if zero_weights:
                weights = np.zeros(weight_shape)
            else:
                weights = np.random.rand(*weight_shape)
            weights_per_layer.append(np.matrix(weights))
        return weights_per_layer

    @staticmethod
    def _eta_generator(eta, eta_init):
        if eta == 'constant':
            return lambda epoch: eta_init
        elif eta == 'decreasing':
            return lambda epoch: eta_init / np.sqrt(epoch)
        else:
            raise ValueError("eta argument should be 'constant' or 'decreasing'")

    def _backpropagate_aux(self, activations_prev_layer, w_num, label, output_grads):
        if w_num == self._num_layers:
            _, cost_grad = self._cost_function(activations_prev_layer[:-1, :], label)
            # cost gradient w.r.t the network output (a). We delete the unnescesary bias term
            return cost_grad

        w = self._weights[w_num - 1]
        z = w.T * activations_prev_layer  # this is the weighted sum of inputs to this layer (including bias term)
        a, a_grad = self._activation_function(z) if w_num < self._num_layers - 1 else (z, 1)  # activation values (inputs to next layer) and its gradient w.r.t z
        a_with_bias = np.vstack((a, 1))
        delta_next = self._backpropagate_aux(a_with_bias, w_num + 1, label, output_grads)  # cost gradient w.r.t a
        delta = w * np.multiply(delta_next, a_grad)  # cost gradient w.r.t activations_prev_layer
        _, regularization_grad = self._regularization_function(w)
        grad_w = activations_prev_layer * np.multiply(delta_next, a_grad).T + regularization_grad
        output_grads.append(grad_w)
        return delta[:-1, :]  # return delta without bias term which does not propagate backwards

    def backpropagate(self, sample, label):
        '''backpropagates given a sample and a label with a stepsize of eta'''
        output_grads = []
        self._backpropagate_aux(np.vstack((sample, 1)), 1, label, output_grads)
        output_grads.reverse()

        if self._check_gradients:
            self._gradient_check(sample, label, self._weights, output_grads)

        for w, grad_w in zip(self._weights, output_grads):
            w -= self._eta_for_epoch(self._epoch) * grad_w

        self._epoch += 1

    def _gradient_check(self, sample, label, w_list, w_grad_list):
        """Gradient check for a function f using symmetric differences numerical gradient
        """
        def f(w, i):
            updated_list = list(w_list)
            updated_list[i] = w
            prediction = sample
            reg = 0
            for w in updated_list[:-1]:
                prediction = np.vstack((prediction, np.ones((1, prediction.shape[1]))))
                prediction, _ = self._activation_function(w.T * prediction)
                reg_w, _ = self._regularization_function(w)
                reg += reg_w
            prediction = np.vstack((prediction, np.ones((1, prediction.shape[1]))))
            prediction = updated_list[-1].T * prediction
            cost, _ = self._cost_function(prediction, label)
            return cost + reg

        EPSILON = 1.e-4

        for i, (w, grad) in enumerate(zip(w_list, w_grad_list)):
            # Iterate over all indexes in x
            it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                orig_x = w[ix]

                w[ix] = orig_x + EPSILON
                f_plus = f(w, i)

                w[ix] = orig_x - EPSILON
                f_minus = f(w, i)

                w[ix] = orig_x

                numgrad = (f_plus - f_minus) / (2. * EPSILON)

                # Compare gradients
                diff = np.max(np.abs((numgrad - grad[ix])))
                if diff > 1e-5:
                    error_message = "Gradient check failed at layer %d.\n" % (i + 1)
                    error_message += "First gradient error found at index %s\n" % str(ix)
                    error_message += "Gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
                    raise NeuralNetwork.GradientError(error_message)

                it.iternext()  # Step to next dimension

    def predict(self, sample):
        """Returns the prediction for the label of the given the sample"""
        prediction = sample
        for w in self._weights[:-1]:
            prediction = np.vstack((prediction, np.ones((1, prediction.shape[1]))))
            prediction, _ = self._activation_function(w.T * prediction)
        prediction = np.vstack((prediction, np.ones((1, prediction.shape[1]))))
        prediction = self._weights[-1].T * prediction
        return prediction



def get_train_and_test_sets():
    dataset = load_boston()     
    
    # normalize the inputs to have std of 1 and mean of 0
    features = ((dataset.data - dataset.data.mean(axis=0)) /
            dataset.data.std(axis=0))
    num_samples = np.size(features, axis=0)
    targets = dataset.target[:, np.newaxis]

    # shuffle examples and targets
    perm = np.random.permutation(num_samples)
    features = features[perm]
    targets = targets[perm]

    train_size = int(num_samples * 8 / 10)
    test1_size = int(num_samples / 10)

    train = features[:train_size, :]
    train_targets = targets[:train_size]     
    test1 = features[train_size:train_size + test1_size, :]
    test1_targets = targets[train_size:train_size + test1_size]
    test2 = features[train_size + test1_size:, :]
    test2_targets = targets[train_size + test1_size:]
    return ((train, train_targets), (test1,test1_targets),
            (test2,test2_targets))


def mse(predictions, targets):
    return np.mean(np.array(predictions - targets) ** 2)


def train_nn(EPOCHS, X_train, X_validation, Y_train, Y_validation, nn):
    train_errors = list()
    validation_errors = list()
    for _ in range(EPOCHS):
        indices = range(len(list(Y_train)))
        np.random.shuffle(indices)  # random permutation
        train_errors.append(mse(np.vstack([nn.predict(x.T) for x in np.matrix(X_train)]), np.matrix(Y_train)))
        validation_errors.append(
            mse(np.vstack([nn.predict(x.T) for x in np.matrix(X_validation)]), np.matrix(Y_validation)))
        for i in indices:
            x = np.matrix(X_train[i, :])
            y = np.matrix(Y_train[i])
            nn.backpropagate(x.T, y)
    return train_errors, validation_errors


def experiment1(train_and_test_sets):
    (train, validation, _) = train_and_test_sets
    X_train, Y_train = train
    X_validation, Y_validation = validation

    indices = range(len(list(Y_train)))
    np.random.shuffle(list(indices))  # random permutation
    nn = NeuralNetwork([13, 50, 1], squared_loss, sigmoid, norm_2_regularization, eta_init=0.01 / len(Y_train))
    train_errors = list()
    validation_errors = list()
    train_errors.append(mse(np.vstack([nn.predict(x.T) for x in np.matrix(X_train)]), np.matrix(Y_train)))
    validation_errors.append(mse(np.vstack([nn.predict(x.T) for x in np.matrix(X_validation)]), np.matrix(Y_validation)))
    for i in indices:
        x = np.matrix(X_train[i, :])
        y = np.matrix(Y_train[i])
        nn.backpropagate(x.T, y)
        train_errors.append(mse(np.vstack([nn.predict(x.T) for x in np.matrix(X_train)]), np.matrix(Y_train)))
        validation_errors.append(mse(np.vstack([nn.predict(x.T) for x in np.matrix(X_validation)]), np.matrix(Y_validation)))

    plt.figure()
    plt.plot(train_errors, 'b', label="Train Error")
    plt.plot(validation_errors, 'r', label="Validation Error")
    plt.title("Layout:{}, $\lambda={}$, $\eta=0.01/m$".format([13, 50, 1], 0))
    ax = plt.gca()
    ax.set_xlabel(r"GD Iteration")
    ax.set_ylabel(r"MSE")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.show()


def experiment2(train_and_test_sets):
    (train, validation, _) = train_and_test_sets
    X_train, Y_train = train
    X_validation, Y_validation = validation
    layout = [13, 50, 1]
    EPOCHS = 600
    eta_init = 0.01 / len(Y_train)
    nn = NeuralNetwork(layout, squared_loss, sigmoid, norm_2_regularization, eta_init=eta_init)

    train_errors, validation_errors = train_nn(EPOCHS, X_train, X_validation, Y_train, Y_validation, nn)

    plt.figure()
    plt.plot(train_errors, 'b', label="Train Error")
    plt.plot(validation_errors, 'r', label="Validation Error")
    plt.title(r"Layout:{}, $\lambda={}$, $\eta=0.01/m$".format(layout, 0))
    ax = plt.gca()
    ax.set_xlabel(r"Epoch")
    ax.set_ylabel(r"MSE")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.show()


def experiment3(train_and_test_sets):
    (train, validation, _) = train_and_test_sets
    X_train, Y_train = train
    X_validation, Y_validation = validation
    layout = [13, 50, 1]
    EPOCHS = 600
    eta_init = 0.01 / len(Y_train)
    nn = NeuralNetwork(layout, squared_loss, sigmoid, norm_2_regularization, 5.0, eta_init=eta_init)

    train_errors, validation_errors = train_nn(EPOCHS, X_train, X_validation, Y_train, Y_validation, nn)

    plt.figure()
    plt.plot(train_errors, 'b', label="Train Error")
    plt.plot(validation_errors, 'r', label="Validation Error")
    plt.title(r"Layout:{}, $\lambda={}$, $\eta=0.01/m$".format(layout, 5.0))
    ax = plt.gca()
    ax.set_xlabel(r"Epoch")
    ax.set_ylabel(r"MSE")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.show()


def experiment4(train_and_test_sets):
    (train, validation, _) = train_and_test_sets
    X_train, Y_train = train
    X_validation, Y_validation = validation
    layout = [13, 50, 20, 1]
    EPOCHS = 600
    eta_init = 0.1

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    f.suptitle(r"Layout:{}, $\eta=0.1/\sqrt{{epoch}}$".format(layout))
    ax1.set_xlabel(r"Epoch")
    ax2.set_xlabel(r"Epoch")
    ax3.set_xlabel(r"Epoch")
    ax1.set_ylabel(r"MSE")

    def train_and_plot(ax, reg_factor):
        nn = NeuralNetwork(layout, squared_loss, sigmoid, norm_2_regularization, reg_factor, eta_init=eta_init,
                           eta='decreasing')
        train_errors, validation_errors = train_nn(EPOCHS, X_train, X_validation, Y_train, Y_validation, nn)
        ax.set_title("$\lambda={}$, min test error: {}".format(reg_factor, np.min(validation_errors)))
        ax.plot(train_errors, 'b', label="Train Error")
        ax.plot(validation_errors, 'r', label="Validation Error")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

    for ax, reg_factor in [(ax1, 0.0), (ax2, 0.1), (ax3, 1.)]:
        train_and_plot(ax, reg_factor)

    plt.show()

    eta_init = 0.01 / len(Y_train)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    f.suptitle(r"Layout:{}, $\eta=0.01/(m*\sqrt{{epoch}})$".format(layout))
    ax1.set_xlabel(r"Epoch")
    ax2.set_xlabel(r"Epoch")
    ax3.set_xlabel(r"Epoch")
    ax1.set_ylabel(r"MSE")
    for ax, reg_factor in [(ax1, 0.0), (ax2, 0.01), (ax3, 0.1)]:
        train_and_plot(ax, reg_factor)

    plt.show()


def experiment5(train_and_test_sets):
    (train, validation, _) = train_and_test_sets
    X_train, Y_train = train
    X_validation, Y_validation = validation
    layout = [13, 50, 20, 1]
    EPOCHS = 600
    eta_init = 0.1
    reg_factor = 0.1

    nn = NeuralNetwork(layout, squared_loss, sigmoid, norm_2_regularization, reg_factor, eta_init=eta_init,
                       eta='decreasing')
    train_errors = list()
    validation_errors = list()
    iterations = 0
    for _ in range(EPOCHS):
        indices = range(len(list(Y_train)))
        np.random.shuffle(indices)  # random permutation
        for i in indices:
            if iterations % 10 == 0:
                train_errors.append(mse(np.vstack([nn.predict(x.T) for x in np.matrix(X_train)]), np.matrix(Y_train)))
                validation_errors.append(
                    mse(np.vstack([nn.predict(x.T) for x in np.matrix(X_validation)]), np.matrix(Y_validation)))
            x = np.matrix(X_train[i, :])
            y = np.matrix(Y_train[i])
            nn.backpropagate(x.T, y)
            iterations += 1

    plt.figure()
    xs = range(0, iterations, 10)
    plt.plot(xs, train_errors, 'b', label="Train Error")
    plt.plot(xs, validation_errors, 'r', label="Validation Error")
    plt.title(r"Layout:{}, $\lambda={}$, $\eta=0.1/\sqrt{{epoch}}$".format(layout, reg_factor))
    ax = plt.gca()
    ax.set_xlabel(r"Gradient Descent Iteration")
    ax.set_ylabel(r"MSE")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.show()


def experiment6(train_and_test_sets):
    (train, validation, test) = train_and_test_sets
    X_train, Y_train = train
    X_validation, Y_validation = validation
    X_test, Y_test = test
    layout = [13, 50, 20, 1]
    EPOCHS = 600
    eta_init = 0.1
    reg_factor = 0.1

    nn = NeuralNetwork(layout, squared_loss, sigmoid, norm_2_regularization, reg_factor, eta_init=eta_init,
                       eta='decreasing')
    train_errors = list()
    validation_errors = list()
    test_errors = list()
    for _ in range(EPOCHS):
        indices = range(len(Y_train))
        np.random.shuffle(indices)  # random permutation
        train_errors.append(mse(np.vstack([nn.predict(x.T) for x in np.matrix(X_train)]), np.matrix(Y_train)))
        validation_errors.append(
            mse(np.vstack([nn.predict(x.T) for x in np.matrix(X_validation)]), np.matrix(Y_validation)))
        test_errors.append(mse(np.vstack([nn.predict(x.T) for x in np.matrix(X_test)]), np.matrix(Y_test)))
        for i in indices:
            x = np.matrix(X_train[i, :])
            y = np.matrix(Y_train[i])
            nn.backpropagate(x.T, y)

    plt.figure()
    plt.plot(train_errors, 'b', label="Train Error")
    plt.plot(validation_errors, 'r', label="Validation Error")
    plt.plot(test_errors, 'g', label="Test Error")
    plt.title(r"Layout:{}, $\lambda={}$, $\eta=0.1/\sqrt{{epoch}}$, min test MSE: {}".format(layout, reg_factor,
                                                                                             np.min(test_errors)))
    ax = plt.gca()
    ax.set_xlabel(r"Epoch")
    ax.set_ylabel(r"MSE")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.show()


def main():
    '''runs all experiments'''
    np.random.seed(54353689)
    train_and_test_sets = get_train_and_test_sets()

    # experiments
    experiment1(train_and_test_sets)
    experiment2(train_and_test_sets)
    experiment3(train_and_test_sets)
    experiment4(train_and_test_sets)
    experiment5(train_and_test_sets)
    experiment6(train_and_test_sets)


if __name__ == '__main__':
    main()
