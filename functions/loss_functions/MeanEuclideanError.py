import numpy as np

from functions.loss_functions.LossFunction import LossFunction

class MeanEuclideanError (LossFunction):

    @staticmethod
    def function(target, predicted):
        square = np.square(np.subtract(target, predicted))
        sum_of_squares = np.sum(square)
        distances = np.sqrt(sum_of_squares)
        return np.average(distances) # questa riga è inutile perché distances è un numero solo


    def first_derivative(self, target, predicted):
        return np.divide(np.subtract(predicted, target),self.function(target,predicted))