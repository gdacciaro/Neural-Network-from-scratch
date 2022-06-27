from functions.activaction_functions.ActivactionFunctionFactory import ActivactionFunctionFactory

""" This class represents a single layer of the MLP """

class Layer:

    def __init__(self, nodes, activation):
        self.nodes = nodes
        self.activation = activation

    def __str__(self):
        return "[n:" + str(self.nodes)+ \
                  ", act=" + str(self.activation) + "]"

    def __repr__(self):
        return self.__str__()

    def getActivationFunction(self):
        return ActivactionFunctionFactory(self.activation)