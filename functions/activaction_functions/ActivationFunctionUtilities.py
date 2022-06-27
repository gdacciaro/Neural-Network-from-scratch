from functions.activaction_functions.ActivactionFunctionFactory import ActivactionFunctionFactory

def generateActivationFunctionsList(listOfActFuns_strings):
    """ This utility method translates a list of string into a list of activation functions """
    callableActivationFunctions = []
    for string in listOfActFuns_strings:
        realActivationFunction = ActivactionFunctionFactory(string)
        callableActivationFunctions.append(realActivationFunction)
    return callableActivationFunctions
