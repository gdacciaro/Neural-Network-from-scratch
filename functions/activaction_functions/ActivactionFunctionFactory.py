from enum import Enum

from functions.activaction_functions.Identity import Identity
from functions.activaction_functions.LeakyReLU import LeakyReLU
from functions.activaction_functions.ReLU import ReLU
from functions.activaction_functions.Sigmoid import Sigmoid
from functions.activaction_functions.TanH import TanH


class ActivactionFunctionFactory(Enum):
    """ This is the factory for activaction functions,
            this design pattern allows us to have a more maintainable code. """

    tanh = 'tanh'
    relu = 'relu'
    sigmoid = 'sigmoid'
    leakyrelu = 'leakyrelu'
    none = 'none'
    identity = 'identity'

    def __init__(self, activationFunctionSelected):
        if activationFunctionSelected == 'tanh':
            self.theFunction = TanH()
        elif activationFunctionSelected == 'relu':
            self.theFunction = ReLU()
        elif activationFunctionSelected == 'sigmoid':
            self.theFunction = Sigmoid()
        elif activationFunctionSelected == 'leakyrelu':
            self.theFunction = LeakyReLU()
        elif activationFunctionSelected == 'identity':
            self.theFunction = Identity()

    def getFunction(self, input):
        return self.theFunction.function(input)

    def getFirstDerivativeFunction(self, input):
        return self.theFunction.first_derivative(input)
