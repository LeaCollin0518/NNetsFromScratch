from vanilla_net import VanillaNet, mse
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import load_digits # MNIST
import pytest

def test_feed_forward():
    nn, datum, target = setup_MattMazur()

    # test results
    actual_result   = nn.feed_forward(datum)
    expected_result = np.array([[0.7569319154399385, 0.7677178798069613]])
    assert((actual_result == expected_result).all())

def test_backprop_single_row():
    nn, datum, target = setup_MattMazur()

    for _ in range(10000):
        nn._backprop(datum, target)
    actual_result = nn.feed_forward(datum)

    epsilon, error =  0.05, mse(actual_result, target)
    assert(error < epsilon)

@pytest.mark.slow
def test_train():
    nn, datum, target = setup_MattMazur()

    data, targets = tuple(map(lambda M: np.repeat(M, 10000, axis=0), (datum, target)))
    nn.train(data, targets, 1000, 10)

    actual_result = nn.feed_forward(datum)

    epsilon, error =  0.05, mse(actual_result, target)
    assert(error < epsilon)

@pytest.mark.very_slow
def test_MNIST():
    # get the data
    digits = load_digits()

    train_data = np.array([img.reshape(64, ) for img in digits.images])

    targets = digits.target.reshape(digits.target.shape[0], 1)
    targets = LabelBinarizer().fit_transform(targets)

    # initialize the NN
    nn_dims = [train_data.shape[1], 100, 50, targets.shape[1]]
    nn = VanillaNet(nn_dims, learning_rate=0.1)

    # train the net. 180 batch of 100 imgs each. 500 epochs
    errs = nn.train(train_data, targets, 180, 500)

    # calculate accuracy
    accuracy = 0
    for img, target in zip(train_data, digits.target):
        prediction = np.argmax(nn.feed_forward(img.reshape(1, 64)))
        if prediction == target: accuracy += 1
    accuracy /= train_data.shape[0]

    assert(accuracy > .99)

def setup_MattMazur():
    """
    Neural net from https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    """
    # initialize nn
    nn = VanillaNet([2,2,2])
    w1 = np.array([[0.15, 0.20], [0.25, 0.30], [0.35, 0.35]])
    w2 = np.array([[0.40, 0.45], [0.5, 0.55], [0.60, 0.60]])
    nn.weights = [w1, w2]

    # initialize datum and target
    datum = np.array([[0.05, 0.10]])
    target = np.array([[0.01, 0.99]])

    return nn, datum, target
