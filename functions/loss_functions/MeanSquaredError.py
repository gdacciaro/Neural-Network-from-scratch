import numpy as np

from functions.loss_functions.LossFunction import LossFunction


# Formula: sum from i=1 to n of 1/n*(abs(target) - abs(predicted[i]))^2
class MeanSquaredError(LossFunction):

    @staticmethod
    def function(target, predicted):
        # Preconditions:
        if target is None:
            raise Exception("null target")
        if predicted is None:
            raise Exception("null predicted")

        return np.sum( np.square(np.subtract(target, predicted)))

    def first_derivative(self, target, predicted):
        return np.subtract(predicted, target)