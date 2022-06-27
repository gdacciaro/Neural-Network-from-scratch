from enum import Enum

from functions.loss_functions.MeanEuclideanError import MeanEuclideanError
from functions.loss_functions.MeanSquaredError import MeanSquaredError

#Use Case:
    # mse = LossFunctionFactory("mse")
    # print(mse.getFunction(4.2,[42]))

class LossFunctionFactory(Enum):
    """ This is the factory for Loss functions,
            this design pattern allows us to have a more maintainable code. """
    mse = 'mse'
    mee = 'mee'

    def __init__(self, lossFunctionSelected):
        if lossFunctionSelected == 'mse':
            self.__theLoss = MeanSquaredError()
            return
        if lossFunctionSelected == 'mee':
            self.__theLoss = MeanEuclideanError()
            return
        raise ValueError("Error")

    def getFunction(self,target,predicted):
        return self.__theLoss.function(target,predicted)

    def getFirstDerivative(self, target, predicted):
        return self.__theLoss.first_derivative(target,predicted)
