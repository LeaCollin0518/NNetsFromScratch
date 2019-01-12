from vanilla_net import VanillaNet
import numpy as np

def test_feed_forward():
    # initialize nn
    nn = VanillaNet([2,2,2])
    w1 = np.array([[0.15, 0.20], [0.25, 0.30], [0.35, 0.35]])
    w2 = np.array([[0.40, 0.45], [0.5, 0.55], [0.60, 0.60]])
    nn.weights = [w1, w2]

    # prepare input
    data = np.array([[0.05, 0.10]])

    # test results
    expected_result = np.array([[0.75693192, 0.76771788]])
    actual_result   = nn.feed_forward(data)
    assert((actual_result == expected_result).all())
