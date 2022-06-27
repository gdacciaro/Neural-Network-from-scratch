import numpy as np

from functions.activaction_functions.ActivactionFunction import ActivactionFunction

class ReLU(ActivactionFunction):

    @staticmethod
    def function(input):
        return np.maximum(0, input)

    @staticmethod
    def first_derivative(input):
        return np.where(input > 0, 1, 0)