import numpy as np
from scipy.special import expit
from collections import deque

class VanillaNet:
    """
    Simple Multi-Layer Perceptron
    """

    def __init__(self, dims, learning_rate=0.01):
        # at least one layer for input and one for output
        assert(len(dims) >= 2)
        # positive learning rate
        assert(learning_rate > 0)

        self.dims = dims
        self._initialize_weights()

        self.learning_rate = learning_rate

    def _initialize_weights(self):
        self.weights = []

        for fst, snd in zip(self.dims, self.dims[1:]):
            weight_matrix = np.random.normal(0.0, 1.0, size=(fst+1, snd)) # +1 for bias
            self.weights.append(weight_matrix)

    def feed_forward(self, datum, for_backprop=False):
        # input data is a row vector (1 x dims[0])
        assert(len(datum.shape) == 2 and datum.shape[0] == 1 and datum.shape[1] == self.dims[0])

        one = np.array([[1]])
        curr = datum

        if for_backprop:
            activations = []

        for weight_matrix in self.weights:
            # adding bias term
            curr = np.concatenate([curr, one], axis=1)

            if for_backprop:
                activations.append(curr)

            # feed forward one layer
            curr = expit(curr.dot(weight_matrix))

        if for_backprop:
            activations.append(curr)
            return activations
        else:
            return curr

    def train(self, data, targets):
        assert(targets.shape[0] == data.shape[0])
        # input data must be matrix (n_rows x dims[0]), n_rows >= 1
        assert(len(data.shape) == 2 and data.shape[1] == self.dims[0])
        # targets must be matrix (n_rows x dims[-1]), n_rows >= 1
        assert(len(targets.shape) == 2 and targets.shape[1] == self.dims[-1])

        n_rows = data.shape[0]

        gradients = deque([np.zeros(weight_matrix.shape) for weight_matrix in self.weights])

        for i in range(n_rows):
            # extract row i of data and target
            datum, target = tuple(map(lambda M: M[i,:].reshape(1, M.shape[1]), (data, targets)))

            activations = self.feed_forward(datum, for_backprop=True)

            # calculate gradients for each layer
            for layer_i in reversed(range(len(self.dims))):
                if layer_i == 0: break # nothing to do for input layer

                if layer_i == len(self.dims)-1:
                    err_wrt_activations = -(target - activations[layer_i])
                    curr_acts = activations[layer_i]
                else:
                    err_wrt_activations = self.weights[layer_i].dot(deltas.T).T[:,:-1]
                    curr_acts = activations[layer_i][:,:-1]

                deltas = err_wrt_activations * (curr_acts * (1 - curr_acts))
                gradients[layer_i-1] += activations[layer_i-1].T.dot(deltas)

        for gradient in gradients:
            gradient /= n_rows # average gradient over all data

        for gradient, weight_matrix in zip(gradients, self.weights):
            weight_matrix -= self.learning_rate * gradient

    @staticmethod
    def calculate_error(output, target):
        return np.sum(0.5*(target - output)**2)
