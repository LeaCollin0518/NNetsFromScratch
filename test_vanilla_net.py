from vanilla_net import VanillaNet
import numpy as np

def test_feed_forward():
    nn, datum = get_testnet_datum()

    # test results
    expected_result = np.array([[0.7569319154399385, 0.7677178798069613]])
    actual_result   = nn.feed_forward(datum)
    assert((actual_result == expected_result).all())

def test_backprop():
    nn, datum = get_testnet_datum()

    expected_result = np.array([[0.01, 0.99]])
    actual_result = np.array([[-0.50, -0.50]]) # TODO: implement backprop functionaliatiy

    epsilon, error =  0.001, VanillaNet.calulate_error(expected_result, actual_result)
    assert(error < epsilon)

def get_testnet_datum():
    # initialize nn
    nn = VanillaNet([2,2,2])
    w1 = np.array([[0.15, 0.20], [0.25, 0.30], [0.35, 0.35]])
    w2 = np.array([[0.40, 0.45], [0.5, 0.55], [0.60, 0.60]])
    nn.weights = [w1, w2]

    # initialize datum
    datum = np.array([[0.05, 0.10]])

    return nn, datum
