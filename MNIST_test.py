#!/usr/bin/env python

import numpy as np
from vanilla_net import VanillaNet
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelBinarizer

digits = load_digits()

# reshaping test data for neural net
train_data = np.array([img.reshape(64, ) for img in digits.images])

targets = digits.target.reshape(digits.target.shape[0], 1)

# one hot encoding of targets
target_lb = LabelBinarizer()
targets = target_lb.fit_transform(targets)

nn = VanillaNet([train_data.shape[1], 100, 50, 10])
nn.learning_rate = 0.1

errs = nn.train(train_data, targets, 180, 500)

accuracy = 0

for i, img in enumerate(train_data):
    image = img.reshape(1, 64)
    prediction = np.argmax(nn.feed_forward(image))
    if prediction == digits.target[i]:
        accuracy += 1

accuracy /= train_data.shape[0]
print (accuracy)
