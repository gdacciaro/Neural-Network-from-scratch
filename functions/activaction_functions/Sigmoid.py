import time

import numpy as np
from scipy.special import logit, expit
from functions.activaction_functions.ActivactionFunction import ActivactionFunction

class Sigmoid(ActivactionFunction):

    @staticmethod
    def function(input):
        #input = np.clip(input, -500, 500)  # Prevent overflow.
        #return 1.0 / (1 + np.exp(-input))
        return expit(input)

    @staticmethod
    def first_derivative(input):
        sigmoid = Sigmoid().function(input)
        derivative = sigmoid * (1 - sigmoid)
        return derivative