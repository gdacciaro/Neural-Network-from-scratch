import numpy as np

from functions.activaction_functions.ActivactionFunction import ActivactionFunction

class LeakyReLU(ActivactionFunction):

    @staticmethod
    def function(input, alpha = 0.0001):
        return np.maximum(input*alpha, input)

    @staticmethod
    def first_derivative(input, alpha = 0.0001):
        return np.where(input >= 0, 1, alpha)