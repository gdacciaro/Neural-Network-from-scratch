from abc import abstractmethod


class LearningRateDecay:

    @staticmethod
    @abstractmethod
    def getNextLearningRate(actual_lr, initial_lr, actual_epoch):
        pass