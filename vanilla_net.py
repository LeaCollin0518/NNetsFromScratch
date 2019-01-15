import numpy as np
from scipy.special import expit

class VanillaNet:
    def __init__(self, dims):
        # at least one layer for input and one for output
        assert(len(dims) >= 2)

        self.dims = dims
        self._initialize_weights()

    def _initialize_weights(self):
        self.weights = []

        for fst, snd in zip(self.dims, self.dims[1:]):
            weight_matrix = np.random.normal(0.0, 1.0, size=(fst+1, snd)) # +1 for bias
            self.weights.append(weight_matrix)

    def feed_forward(self, datum):
        #  input data is a row vector (1 x dims[0])
        assert(len(datum.shape) == 2 and datum.shape[0] == 1 and datum.shape[1] == self.dims[0])

        one = np.array([[1]])
        curr = datum

        for weight_matrix in self.weights:
            # adding bias term
            curr = np.concatenate([curr, one], axis=1)

            # feed forward one layer
            curr = expit(curr.dot(weight_matrix))

        return curr

    @staticmethod
    def calculate_error(target, output):
        return np.sum(0.5*(target - output)**2)

