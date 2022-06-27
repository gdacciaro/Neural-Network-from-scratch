import numpy as np

from functions.activaction_functions.ActivactionFunction import ActivactionFunction

class TanH(ActivactionFunction):

    @staticmethod
    def function(input):
        return np.tanh(input)

    @staticmethod
    def first_derivative(input):
        t = np.tanh(input)
        dt = np.square(1 - t)
        return dt