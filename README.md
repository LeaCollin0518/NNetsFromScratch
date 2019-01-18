# Neural Networks From Scratch

## VanillaNet in `vanilla_net.py` and `test_vanilla_net.py`

`vanilla_net.py` contains a `numpy`-only implementation of multi-layer perceptron. This neural net implementation is based on [this article](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/) and tested partially based on data and targets from that article as well.

`test_vanilla_net.py` contains tests for the neural net. Two of the tests are 'marked' with `pytest.mark.slow` and `pytest.mark.very_slow`, respectively. To run all tests **except** for the `slow` and `very_slow` ones, run

```
$ pytest -m "not slow and not very_slow"
```

Otherwise, to run all tests, just run

```
$ pytest
```
