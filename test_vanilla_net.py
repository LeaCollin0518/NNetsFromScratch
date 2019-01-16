from vanilla_net import VanillaNet, mse
import numpy as np

def test_feed_forward():
    nn, datum, target = setup()

    # test results
    actual_result   = nn.feed_forward(datum)
    expected_result = np.array([[0.7569319154399385, 0.7677178798069613]])
    assert((actual_result == expected_result).all())

def test_backprop_single_row():
    nn, datum, target = setup()

    for _ in range(10000):
        nn._backprop(datum, target)
    actual_result = nn.feed_forward(datum)

    epsilon, error =  0.05, mse(actual_result, target)
    assert(error < epsilon)

# data,targets = tuple(map(lambda row: np.repeat(row, 10000, axis=0), (datum,target)))

def test_train():
    nn, datum, target = setup()

    data, targets = tuple(map(lambda M: np.repeat(M, 10000, axis=0), (datum, target)))
    nn.train(data, targets, 1000, 10)

    actual_result = nn.feed_forward(datum)

    epsilon, error =  0.05, mse(actual_result, target)
    assert(error < epsilon)

def setup():
    # initialize nn
    nn = VanillaNet([2,2,2])
    w1 = np.array([[0.15, 0.20], [0.25, 0.30], [0.35, 0.35]])
    w2 = np.array([[0.40, 0.45], [0.5, 0.55], [0.60, 0.60]])
    nn.weights = [w1, w2]

    # initialize datum and target
    datum = np.array([[0.05, 0.10]])
    target = np.array([[0.01, 0.99]])

    return nn, datum, target
