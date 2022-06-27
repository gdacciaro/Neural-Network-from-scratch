import math

from functions.lr_decay_functions.LearningRateDecay import LearningRateDecay


class LearningRateDecay_Exponential(LearningRateDecay):

    @staticmethod
    def getNextLearningRate(actual_lr, initial_lr, actual_epoch):

        k = 0.01
        return initial_lr * math.exp(-k * actual_epoch)
