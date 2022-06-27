from abc import abstractmethod

class LossFunction:

    @staticmethod
    @abstractmethod
    def function(target, predicted):
        pass

    @staticmethod
    @abstractmethod
    def first_derivative(target, predicted):
        pass
