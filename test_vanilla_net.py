from vanilla_net import VanillaNet
import numpy as np

def test_feed_forward():
    nn, datum = get_testnet_datum()

    # test results
    expected_result = np.array([[0.7569319154399385, 0.7677178798069613]])
    actual_result   = nn.feed_forward(datum)
    assert((actual_result == expected_result).all())

def test_backprop_single_row():
    nn, datum = get_testnet_datum()

    expected_result = np.array([[0.01, 0.99]])

    for _ in range(10000):
        nn.train(datum, expected_result)
    actual_result = nn.feed_forward(datum)

    epsilon, error =  0.05, VanillaNet.calculate_error(actual_result, expected_result)
    assert(error < epsilon)

# data,targets = tuple(map(lambda row: np.repeat(row, 200000, axis=0), (datum,target)))

def test_backprop_100_rows():
    nn, datum = get_testnet_datum()

    expected_result = np.array([[0.01, 0.99]])
    actual_result = np.array([[-0.50, -0.50]]) # TODO: implement backprop functionaliatiy

def get_testnet_datum():
    # initialize nn
    nn = VanillaNet([2,2,2])
    w1 = np.array([[0.15, 0.20], [0.25, 0.30], [0.35, 0.35]])
    w2 = np.array([[0.40, 0.45], [0.5, 0.55], [0.60, 0.60]])
    nn.weights = [w1, w2]

    # initialize datum
    datum = np.array([[0.05, 0.10]])

    return nn, datum
