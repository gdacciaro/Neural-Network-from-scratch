from functions.activaction_functions.ActivactionFunction import ActivactionFunction

class Identity(ActivactionFunction):

    @staticmethod
    def function(input):
        return input

    @staticmethod
    def first_derivative(input):
        return 1

